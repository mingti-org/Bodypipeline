"""Offline, journaled regrouping. Sealed manifests and data bytes stay immutable."""

from __future__ import annotations

import os
from pathlib import Path

from pipeline.shared.collections import collection_metadata, dated_batch
from pipeline.shared.core import DomainError, atomic_json, digest, file_hash, fsync_dir

TERMINAL_JOBS = {
    "COMPLETED",
    "CONVERTED",
    "SYNTHETIC_READY",
    "QUARANTINED",
    "FAILED",
    "INTERRUPTED",
    "BLOCKED",
    "CANCELLED",
    "TRASHED",
    "DELETED",
}


def _replace_paths(value, replacements):
    if isinstance(value, dict):
        return {k: _replace_paths(v, replacements) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace_paths(v, replacements) for v in value]
    if isinstance(value, str):
        for old, new in replacements.items():
            if value == old or value.startswith(old + "/"):
                return new + value[len(old) :]
    return value


def _verify_files(root, specs):
    for spec in specs:
        path = root / spec["path"]
        if path.resolve() != path or not path.is_file():
            raise DomainError(f"迁移源文件不可用：{path}")
        if (
            path.stat().st_size != spec["size_bytes"]
            or file_hash(path) != spec["sha256"]
        ):
            raise DomainError(f"迁移源文件校验失败：{path}")


def _related_jobs(store, selected):
    jobs = {
        row["id"]: row
        for row in store.all("job")
    }

    def source_uploads(job, seen=None):
        seen = set() if seen is None else seen
        if job["id"] in seen:
            return set()
        seen.add(job["id"])
        uploads = set(job.get("upload_ids") or [])
        if job.get("upload_id"):
            uploads.add(job["upload_id"])
        for key in ("source_job_id",):
            identifier = job.get(key)
            if identifier in jobs:
                uploads.update(source_uploads(jobs[identifier], seen))
        for identifier in job.get("source_job_ids") or []:
            if identifier in jobs:
                uploads.update(source_uploads(jobs[identifier], seen))
        for source in job.get("merge_sources") or []:
            if not isinstance(source, dict):
                continue
            identifier = source.get("job_id") or source.get("id")
            if identifier in jobs:
                uploads.update(source_uploads(jobs[identifier], seen))
        return uploads

    return {
        job_id: (job, source_uploads(job))
        for job_id, job in jobs.items()
        if selected.intersection(source_uploads(job))
    }


def plan_split(ingest, assignments, *, verify_files=True):
    """Assignments are reviewed Grus receipts, keyed by stable upload ID."""
    if not assignments or len({a["upload_id"] for a in assignments}) != len(
        assignments
    ):
        raise DomainError("迁移清单为空或包含重复 upload_id")
    rows = {row["id"]: row for row in ingest.store.all("upload")}
    entries, replacements, source_ids = [], {}, set()
    for a in assignments:
        row = rows.get(a["upload_id"])
        if not row or row["state"] != "COMMITTED" or row.get("imported"):
            raise DomainError("迁移仅接受已提交的原始上传")
        if row.get("collection_migration"):
            raise DomainError("该采集已有批次迁移记录，请先检查现有迁移日志")
        if (
            row["session_id"] != a["session_id"]
            or row["manifest_hash"] != a["manifest_hash"]
        ):
            raise DomainError("Grus 回执与服务器会话或清单摘要不一致")
        manifest = ingest.manifest(row, fresh=True)
        if digest(manifest) != row["manifest_hash"]:
            raise DomainError("服务器清单摘要不一致")
        old = collection_metadata(manifest, row["target"], row["prefix"])
        if (
            row.get("collection_id") != old["collection_id"]
            or old["collection_id"] != a["original_collection_id"]
        ):
            raise DomainError("原始 collection 与迁移清单不一致")
        batch = dated_batch(a["collection_date"])
        after = {
            **row,
            **collection_metadata({**manifest, **batch}, row["target"], row["prefix"]),
        }
        if after["collection_id"] == row["collection_id"]:
            raise DomainError("源批次已经使用目标分组")
        if (
            a.get("new_collection_id")
            and a["new_collection_id"] != after["collection_id"]
        ):
            raise DomainError("两端计算出的目标 collection 不一致")
        old_dir, new_dir = Path(row["raw_path"]).parent, ingest.final_dir(after)
        if old_dir != ingest.final_dir(row) or old_dir.resolve() != old_dir:
            raise DomainError("源上传不使用受管 collection 布局")
        if new_dir.exists() or new_dir.resolve() != new_dir:
            raise DomainError("目标路径已存在或包含符号链接")
        if verify_files:
            _verify_files(old_dir / "data", manifest["files"])
        after.update(
            raw_path=str(new_dir / "data"),
            collection_migration={
                "version": 1,
                "original_collection_id": row["collection_id"],
                "manifest_hash": row["manifest_hash"],
                "original_raw_path": row["raw_path"],
            },
        )
        source_ids.add(row["collection_id"])
        replacements[str(old_dir)] = str(new_dir)
        entries.append(
            {
                "before": row,
                "after": after,
                "old_dir": str(old_dir),
                "new_dir": str(new_dir),
                "files": manifest["files"],
            }
        )
    selected = {a["upload_id"] for a in assignments}
    actual = {
        r["id"]
        for r in rows.values()
        if r.get("collection_id") in source_ids and r["state"] == "COMMITTED"
    }
    if selected != actual:
        raise DomainError(
            "迁移清单必须覆盖源 collection 的全部已提交采集；请重新核对数量"
        )
    by_id = {e["after"]["id"]: e["after"] for e in entries}
    state_updates, file_updates = [], []
    related_jobs = _related_jobs(ingest.store, selected)
    # Update references through Store.put, so Store implementations can invalidate
    # their indexes. Completed conversion outputs keep their original locations.
    for kind_dir in sorted(ingest.store.path.iterdir()):
        if not kind_dir.is_dir() or kind_dir.name == "upload":
            continue
        for before in ingest.store.all(kind_dir.name):
            related = (
                related_jobs.get(before.get("id"), (None, set()))[1]
                if kind_dir.name == "job"
                else set()
            )
            touched = selected.intersection(related)
            if touched and kind_dir.name == "job":
                if before.get("state") not in TERMINAL_JOBS:
                    raise DomainError("相关转换任务尚未结束，不能迁移其原始数据")
            after = _replace_paths(before, replacements)
            if touched and kind_dir.name == "job":
                after["source_collection_ids"] = sorted(
                    {
                        by_id[uid]["collection_id"]
                        if uid in by_id
                        else rows[uid]["collection_id"]
                        for uid in related
                    }
                )
                after["collection_migration_note"] = (
                    "原始采集已按日期分批；此转换结果保留原提交快照和数据路径"
                )
                dataset = before.get("dataset_path")
                if dataset:
                    lineage = Path(dataset).parent / "lineage.json"
                    if lineage.is_file():
                        import json

                        content = json.loads(lineage.read_text())
                        updated = _replace_paths(content, replacements)
                        updated["source_collection_ids"] = after[
                            "source_collection_ids"
                        ]
                        if updated != content:
                            file_updates.append(
                                {
                                    "path": str(lineage),
                                    "before": content,
                                    "after": updated,
                                }
                            )
            if after != before:
                state_updates.append(
                    {"kind": kind_dir.name, "before": before, "after": after}
                )
    plan = {
        "version": 1,
        "entries": entries,
        "state_updates": state_updates,
        "file_updates": file_updates,
        "assignments": assignments,
    }
    return {**plan, "plan_hash": digest(plan)}


def _check_plan(plan):
    if plan.get("plan_hash") != digest(
        {k: v for k, v in plan.items() if k != "plan_hash"}
    ):
        raise DomainError("迁移计划已修改")


def apply_split(ingest, plan, journal_path):
    """Caller must hold the server ProcessLock (the service must be stopped)."""
    _check_plan(plan)
    journal_path = Path(journal_path)
    if journal_path.exists():
        raise DomainError("迁移日志已存在；请核对完成状态或执行回滚")
    # Rebuild the full plan under the exclusive service lock before writing.
    fresh = plan_split(ingest, plan["assignments"], verify_files=True)
    if fresh != plan:
        raise DomainError("计划生成后源数据或关联记录变化，请重新生成计划")
    journal = {"state": "APPLYING", "plan": plan}
    atomic_json(journal_path, journal)
    try:
        for entry in plan["entries"]:
            old, new = Path(entry["old_dir"]), Path(entry["new_dir"])
            new.parent.mkdir(parents=True, exist_ok=True)
            os.rename(old, new)
            fsync_dir(old.parent)
            fsync_dir(new.parent)
            ingest.store.put("upload", entry["after"]["id"], entry["after"])
        for update in plan["state_updates"]:
            ingest.store.put(update["kind"], update["after"]["id"], update["after"])
        for update in plan["file_updates"]:
            atomic_json(Path(update["path"]), update["after"])
        for entry in plan["entries"]:
            _verify_files(Path(entry["new_dir"]) / "data", entry["files"])
            if (
                digest(ingest.manifest(entry["after"], fresh=True))
                != entry["before"]["manifest_hash"]
            ):
                raise DomainError("迁移后清单摘要变化")
        journal["state"] = "COMPLETED"
        atomic_json(journal_path, journal)
    except BaseException:
        rollback_split(ingest, journal_path)
        raise
    return journal


def rollback_split(ingest, journal_path):
    """Restore from the durable journal, including an interrupted rename/write."""
    import json

    path = Path(journal_path)
    journal = json.loads(path.read_text())
    plan = journal["plan"]
    _check_plan(plan)
    # Refuse to overwrite subsequent changes. This rollback is for the same
    # offline maintenance window, before collection resumes.
    for entry in plan["entries"]:
        current = ingest.store.get("upload", entry["before"]["id"])
        if current not in (entry["before"], entry["after"]):
            raise DomainError("迁移后上传记录发生变化，拒绝覆盖")
        old, new = Path(entry["old_dir"]), Path(entry["new_dir"])
        if old.exists() == new.exists():
            raise DomainError("迁移路径状态不明确，拒绝自动回滚")
    for update in plan["state_updates"]:
        if ingest.store.get(update["kind"], update["before"]["id"]) not in (
            update["before"],
            update["after"],
        ):
            raise DomainError("迁移后关联记录发生变化，拒绝覆盖")
    for update in plan["file_updates"]:
        if json.loads(Path(update["path"]).read_text()) not in (
            update["before"],
            update["after"],
        ):
            raise DomainError("迁移后血缘记录发生变化，拒绝覆盖")
    for update in reversed(plan["file_updates"]):
        atomic_json(Path(update["path"]), update["before"])
    for update in reversed(plan["state_updates"]):
        ingest.store.put(update["kind"], update["before"]["id"], update["before"])
    for entry in reversed(plan["entries"]):
        old, new = Path(entry["old_dir"]), Path(entry["new_dir"])
        if new.exists():
            old.parent.mkdir(parents=True, exist_ok=True)
            os.rename(new, old)
            fsync_dir(old.parent)
            fsync_dir(new.parent)
        ingest.store.put("upload", entry["before"]["id"], entry["before"])
    journal["state"] = "ROLLED_BACK"
    atomic_json(path, journal)
    return journal
