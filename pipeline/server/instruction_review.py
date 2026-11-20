"""Server-owned drafts for in-place LeRobot language-instruction edits."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from pipeline.shared.core import DomainError, digest, executable_path, now

from .datasets import dataset_for_job
from .instruction_templates import (
    default_selections,
    migration_warnings,
    normalize_template,
    render_instruction,
)
from .merges import source_snapshot
from .recycle import RecycleBin, dataset_lock, dataset_read_lock, record_cleanup
from .viewer_data import PreviewRegistry


class InstructionVersion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: StrictInt = Field(ge=0)
    source_signature: str = Field(min_length=64, max_length=64)


class TemplateSaveRequest(InstructionVersion):
    template: dict[str, Any]


class EpisodeSaveRequest(InstructionVersion):
    selections: dict[str, str] = Field(max_length=100)


class ApplyInstructionsRequest(InstructionVersion):
    pass


class InstructionReview:
    def __init__(self, settings, store):
        self.settings, self.store = settings, store
        self.registry = PreviewRegistry(store, settings.viewer_ttl)
        self.resolver = RecycleBin(settings, store, self.registry)

    def resolve(self, dataset_id):
        key, path = self.resolver.resolve(f"bp/{dataset_id}")
        job = self.store.get("job", dataset_id)
        if dataset_for_job(self.settings, job) != path:
            raise DomainError("预览与已登记数据集不匹配，请重新打开预览", 409)
        return key, path, job

    def _source(self, path: Path, snapshot: dict[str, Any]):
        key = digest(
            {"path": str(path), "snapshot": snapshot["signature"], "reader": 1}
        )
        try:
            return self.store.get("instruction_source", key)["data"]
        except DomainError as exc:
            if exc.code != 404:
                raise
        tool = executable_path(self.settings.letools_bin)
        python = Path(tool).with_name("python") if tool else None
        if python is None or not python.is_file():
            raise DomainError("编辑语言指令需要 LeTools Python 环境", 503)
        try:
            result = subprocess.run(
                [
                    str(python),
                    str(Path(__file__).with_name("instruction_writer_worker.py")),
                    "read",
                    str(path),
                ],
                capture_output=True,
                text=True,
                timeout=self.settings.command_timeout,
                env=self.settings.tool_environment(),
            )
        except subprocess.TimeoutExpired as exc:
            raise DomainError("读取语言指令超时，请稍后重试", 504) from exc
        if result.returncode:
            raise DomainError("无法读取语言指令：" + result.stderr[-1200:], 409)
        try:
            data = json.loads(result.stdout)
        except ValueError as exc:
            raise DomainError("语言指令读取器未返回有效结果", 502) from exc
        if source_snapshot(path)["signature"] != snapshot["signature"]:
            raise DomainError("读取期间来源发生变化，请重新加载", 409)
        self.store.put("instruction_source", key, {"data": data})
        return data

    @staticmethod
    def _default_template(data):
        options = []
        for episode in data["episodes"]:
            instruction = episode.get("instruction")
            if isinstance(instruction, str) and instruction.strip() and instruction not in options:
                options.append(instruction.strip())
        if not options:
            options = ["请完成操作"]
        return normalize_template(
            {
                "parts": [{"type": "variable", "key": "instruction"}],
                "variables": {
                    "instruction": {
                        "label": "指令",
                        "options": options,
                        "default": options[0],
                    }
                },
            }
        )

    @staticmethod
    def _snapshot_matches(left, right):
        return (
            left.get("signature") == right.get("signature")
            and left.get("bytes") == right.get("bytes")
        )

    def _rebase_manual_trim(self, row, path, job, snapshot):
        """Carry a saved draft across a BodyPipeline-owned in-place publication."""

        published = job.get("published_dataset_snapshot") or {}
        history = job.get("manual_trim_history")
        if (
            not isinstance(history, list)
            or not history
            or not self._snapshot_matches(snapshot, published)
        ):
            return None
        latest = history[-1]
        if not isinstance(latest, dict) or not latest.get("job_id"):
            return None
        old_signature = row["source_signature"]
        row.update(
            dataset_path=str(path),
            source_signature=snapshot["signature"],
            revision=row["revision"] + 1,
            updated_at=now(),
            last_manual_trim_rebase={
                "job_id": latest["job_id"],
                "old_source_signature": old_signature,
                "source_signature": snapshot["signature"],
                "at": now(),
            },
        )
        self._publish(row)
        return row

    @staticmethod
    def _cleanup_mapping(mapping, data):
        if not isinstance(mapping, dict):
            return None
        normalized = {}
        for raw_old, new in mapping.items():
            try:
                old = int(raw_old)
            except (TypeError, ValueError):
                return None
            if (
                str(old) != raw_old
                or isinstance(new, bool)
                or not isinstance(new, int)
            ):
                return None
            normalized[str(old)] = new
        current = {item["episode_index"] for item in data["episodes"]}
        mapped = set(normalized.values())
        if current != mapped or current != set(range(len(current))):
            return None
        return normalized

    def _rebase_recycle_cleanup(self, key, row, path, snapshot, data):
        """Carry a draft across a completed BodyPipeline permanent cleanup."""

        try:
            recycle = self.store.get("recycle", key)
        except DomainError as exc:
            if exc.code != 404:
                raise
            return None
        if Path(recycle.get("dataset_path", "")) != path:
            return None
        operation = recycle.get("operation") or {}
        if operation.get("state") != "completed" or "mapping" not in operation:
            return None

        # Recover the journal locally. Older deployments may have completed the
        # operation without writing cleanup_history yet.
        journal = {
            **recycle,
            "cleanup_history": [
                dict(item) for item in recycle.get("cleanup_history") or []
            ],
        }
        record_cleanup(journal)
        last_cleanup = (row.get("last_recycle_rebase") or {}).get("cleanup_id")
        for cleanup in reversed(journal["cleanup_history"]):
            if cleanup.get("id") == last_cleanup:
                continue
            mapping = self._cleanup_mapping(cleanup.get("mapping"), data)
            if mapping is None:
                continue
            has_from = isinstance(cleanup.get("from_signature"), str)
            has_to = isinstance(cleanup.get("to_signature"), str)
            if has_from != has_to:
                continue
            if has_from:
                if (
                    cleanup["from_signature"] != row["source_signature"]
                    or cleanup["to_signature"] != snapshot["signature"]
                ):
                    continue
            else:
                updated_at = row.get("updated_at")
                started_at = cleanup.get("started_at")
                if cleanup.get("id") == operation.get("id"):
                    started_at = started_at or operation.get("started_at")
                if (
                    not isinstance(updated_at, (int, float))
                    or isinstance(updated_at, bool)
                    or not isinstance(started_at, (int, float))
                    or isinstance(started_at, bool)
                    or updated_at >= started_at
                ):
                    continue

            episodes, removed, remapped = {}, [], []
            for raw_episode, selections in row["episodes"].items():
                old = int(raw_episode)
                if str(old) not in mapping:
                    removed.append(old)
                    continue
                new = mapping[str(old)]
                episodes[str(new)] = selections
                if new != old:
                    remapped.append([old, new])
            timestamp = now()
            old_signature = row["source_signature"]
            row.update(
                dataset_path=str(path),
                source_signature=snapshot["signature"],
                revision=row["revision"] + 1,
                updated_at=timestamp,
                episodes=episodes,
                last_recycle_rebase={
                    "cleanup_id": cleanup["id"],
                    "old_source_signature": old_signature,
                    "source_signature": snapshot["signature"],
                    "mapping": mapping,
                    "removed_episodes": sorted(removed),
                    "remapped_episodes": sorted(remapped),
                    "at": timestamp,
                },
            )
            self._publish(row)
            return row
        return None

    def _draft(self, key, path, snapshot, data, job=None):
        try:
            row = self.store.get("instruction_draft", key)
        except DomainError as exc:
            if exc.code != 404:
                raise
            return {
                "id": key,
                "dataset_path": str(path),
                "source_signature": snapshot["signature"],
                "revision": 0,
                "template": self._default_template(data),
                "episodes": {},
            }
        if row.get("source_signature") != snapshot["signature"]:
            rebased = self._rebase_recycle_cleanup(key, row, path, snapshot, data)
            if rebased is not None:
                return rebased
            if job is not None:
                rebased = self._rebase_manual_trim(row, path, job, snapshot)
                if rebased is not None:
                    return rebased
            raise DomainError(
                "来源数据已变化，指令草稿不能自动套用；请先核对来源与历史草稿",
                409,
            )
        return row

    @staticmethod
    def _check(row, body):
        if (
            row["revision"] != body.revision
            or row["source_signature"] != body.source_signature
        ):
            raise DomainError(
                "草稿版本或来源已变化，请重新加载后再编辑（不会覆盖其他窗口）",
                409,
            )

    @staticmethod
    def _check_source(row, body):
        if row["source_signature"] != body.source_signature:
            raise DomainError(
                "来源数据已变化，请重新加载后再编辑（不会覆盖其他窗口）",
                409,
            )

    @staticmethod
    def _episode(data, episode):
        for item in data["episodes"]:
            if item["episode_index"] == episode:
                return item
        raise DomainError("Episode 不存在", 404)

    def _summary(self, dataset_id, key, path, snapshot, data, row):
        warnings = migration_warnings(row["template"], row["episodes"])
        return {
            "dataset_id": dataset_id,
            "dataset_key": key,
            "dataset_path": str(path),
            "fps": data["fps"],
            "revision": row["revision"],
            "source_signature": row["source_signature"],
            "template": row["template"],
            "episodes": [
                {
                    "episode_index": item["episode_index"],
                    "frames": item["frames"],
                    "current_instruction": item.get("instruction"),
                    "assigned": str(item["episode_index"]) in row["episodes"],
                }
                for item in data["episodes"]
            ],
            "assignment_count": len(row["episodes"]),
            "warnings": warnings,
            "writable": not warnings,
        }

    def dataset(self, dataset_id):
        key, path, job = self.resolve(dataset_id)
        with dataset_read_lock(self.store, key):
            key, path, job = self.resolve(dataset_id)
            snapshot = source_snapshot(path)
            data = self._source(path, snapshot)
            row = self._draft(key, path, snapshot, data, job)
            return self._summary(dataset_id, key, path, snapshot, data, row)

    def episode(self, dataset_id, episode):
        key, path, job = self.resolve(dataset_id)
        with dataset_read_lock(self.store, key):
            key, path, job = self.resolve(dataset_id)
            snapshot = source_snapshot(path)
            data = self._source(path, snapshot)
            source_episode = self._episode(data, episode)
            row = self._draft(key, path, snapshot, data, job)
            has_draft = str(episode) in row["episodes"]
            selections = row["episodes"].get(str(episode))
            if selections is None:
                selections = default_selections(row["template"])
                current = source_episode.get("instruction")
                if "instruction" in row["template"]["variables"] and current in row["template"]["variables"]["instruction"]["options"]:
                    selections["instruction"] = current
            error = None
            try:
                rendered = render_instruction(row["template"], selections)
                valid = True
            except ValueError as exc:
                rendered, error, valid = None, str(exc), False
            return {
                "episode_index": episode,
                "frames": source_episode["frames"],
                "current_instruction": source_episode.get("instruction"),
                "task_index": source_episode.get("task_index"),
                "revision": row["revision"],
                "source_signature": row["source_signature"],
                "has_draft": has_draft,
                "selections": selections,
                "rendered": rendered,
                "valid": valid,
                "error": error,
            }

    def save_template(self, dataset_id, body: TemplateSaveRequest):
        key, path, job = self.resolve(dataset_id)
        try:
            template = normalize_template(body.template)
        except ValueError as exc:
            raise DomainError(str(exc), 422) from exc
        with dataset_lock(self.store, key):
            key, path, job = self.resolve(dataset_id)
            snapshot = source_snapshot(path)
            data = self._source(path, snapshot)
            row = self._draft(key, path, snapshot, data, job)
            self._check(row, body)
            row["template"] = template
            row.update(revision=row["revision"] + 1, updated_at=now())
            self._publish(row)
            return {
                "revision": row["revision"],
                "source_signature": row["source_signature"],
                "template": template,
                "warnings": migration_warnings(template, row["episodes"]),
                "saved": True,
            }

    def save_episode(self, dataset_id, episode, body: EpisodeSaveRequest):
        key, path, job = self.resolve(dataset_id)
        with dataset_lock(self.store, key):
            key, path, job = self.resolve(dataset_id)
            snapshot = source_snapshot(path)
            data = self._source(path, snapshot)
            self._episode(data, episode)
            row = self._draft(key, path, snapshot, data, job)
            self._check(row, body)
            try:
                rendered = render_instruction(row["template"], body.selections)
            except ValueError as exc:
                raise DomainError(str(exc), 422) from exc
            row["episodes"][str(episode)] = body.selections
            row.update(revision=row["revision"] + 1, updated_at=now())
            self._publish(row)
            return {
                "episode_index": episode,
                "revision": row["revision"],
                "source_signature": row["source_signature"],
                "selections": body.selections,
                "rendered": rendered,
                "saved": True,
            }

    def _publish(self, row):
        history_id = digest(
            {
                "draft": row["id"],
                "revision": row["revision"],
                "signature": row["source_signature"],
            }
        )
        history = {
            **row,
            "id": history_id,
            "draft_id": row["id"],
        }
        self.store.put("instruction_revision", history_id, history)
        self.store.put("instruction_draft", row["id"], row)

    def apply_plan(
        self,
        dataset_id,
        body: ApplyInstructionsRequest,
        episode_indices: set[int] | None = None,
        frozen_assignments: dict[int, str] | None = None,
    ):
        """Called while the apply runner owns the dataset lock."""

        key, path, job = self.resolve(dataset_id)
        snapshot = source_snapshot(path)
        data = self._source(path, snapshot)
        row = self._draft(key, path, snapshot, data, job)
        if frozen_assignments is None:
            self._check(row, body)
        else:
            self._check_source(row, body)
        if not row["episodes"]:
            raise DomainError("请先至少保存一个 Episode 的语言指令", 422)
        selected = set(map(int, row["episodes"]))
        if episode_indices is not None:
            selected &= episode_indices
            if not selected:
                raise DomainError("当前 Episode 尚未保存语言指令", 422)
        assignments = {}
        for raw_episode, selections in sorted(
            row["episodes"].items(), key=lambda item: int(item[0])
        ):
            episode = int(raw_episode)
            if episode not in selected:
                continue
            self._episode(data, episode)
            try:
                assignments[episode] = {
                    "selections": selections,
                    "instruction": render_instruction(row["template"], selections),
                }
            except ValueError as exc:
                raise DomainError(
                    f"Episode {episode} 的指令无效：{exc}", 422
                ) from exc
        rendered_assignments = {
            episode: item["instruction"]
            for episode, item in assignments.items()
        }
        if (
            frozen_assignments is not None
            and rendered_assignments != frozen_assignments
        ):
            raise DomainError(
                "排队期间目标 Episode 指令已变化，请重新提交应用", 409
            )
        return key, path, job, snapshot, row, assignments

    def finalize_apply(self, key, expected_signature, report):
        row = self.store.get("instruction_draft", key)
        snapshot = source_snapshot(Path(row["dataset_path"]))
        operation_id = report.get("operation_id")
        if (
            row["source_signature"] != expected_signature
            and not (
                operation_id
                and (row.get("last_apply_report") or {}).get("operation_id")
                == operation_id
            )
        ):
            raise DomainError("应用期间指令草稿已变化，请重新加载", 409)
        row.update(
            source_signature=snapshot["signature"],
            revision=row["revision"] + 1,
            updated_at=now(),
            last_applied_at=now(),
            last_apply_report=report,
        )
        self._publish(row)
        return snapshot, row
