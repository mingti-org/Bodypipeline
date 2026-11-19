"""Read-only workbench indexes. Mutations always use the authoritative Store."""

from __future__ import annotations

import copy
import inspect
import json
import threading
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import Future

from pipeline.shared.collections import visible_collection_uploads
from pipeline.shared.core import DomainError, digest

from .dataset_management import DELETION_STATES
from .datasets import DATA_READY_STATES, batch_name_for_job, dataset_for_job

UNKNOWN_ROBOT = "__unknown__"
INDEX_KINDS = (
    "upload",
    "job",
    "check",
    "annotation",
    "dataset_deletion",
    "recycle",
    "dataset_description",
)
ACTIVE = {"QUEUED", "RUNNING"}
JOB_SUMMARY_OMIT = frozenset(
    {
        "robot_report",
        "alignment_report",
        "episode_diagnostics",
        "source_manifest_hashes",
        "profile",
        "accepted_capture_profile_hashes",
        "doctor_report",
        "quality_gate",
        "upload_ids",
        "trim_report",
        "trim_policy",
        "merge_sources",
        "source_description",
        "source_snapshot",
    }
)
UPLOAD_LIST_FIELDS = frozenset(
    {
        "id",
        "state",
        "runtime_id",
        "robot_type",
        "robot_id",
        "capture_mode",
        "created_at",
        "total_bytes",
        "file_count",
        "import_path",
        "source_kind",
        "conversion_blockers",
        "collection_id",
        "dataset_name",
        "collection_date",
        "batch_id",
    }
)


def record_robot(row):
    return (
        row.get("robot_type")
        or (row.get("profile") or {}).get("robot_type")
        or UNKNOWN_ROBOT
    )


def ready(row):
    return (
        not (row.get("operation") == "manual_trim" and row.get("in_place") is True)
        and row.get("state") not in DELETION_STATES
        and (row.get("state") in DATA_READY_STATES or row.get("dataset_ready") is True)
    )


class RecordIndex:
    """Refresh local writes by ID; periodically reconcile other processes' writes."""

    def __init__(self, store, robot_resolver=record_robot, ttl=30):
        self.store, self.robot_resolver, self.ttl = store, robot_resolver, ttl
        self.lock = threading.RLock()
        self.changes_lock = threading.Lock()
        self.dirty = defaultdict(set)
        self.rows = {}
        self.by_robot = {}
        self.robots_by_id = {}
        self.versions = defaultdict(int)
        self.robot_versions = defaultdict(int)
        self.tokens = defaultdict(int)
        self.refreshed = {}
        store.subscribe(self.changed)

    def changed(self, kind, identifier):
        # A state writer must never wait for a slow display scan.
        if kind not in INDEX_KINDS:
            return
        with self.changes_lock:
            self.dirty[kind].add(identifier)

    def _sync(self, kind):
        timestamp = time.monotonic()
        with self.changes_lock:
            changed = self.dirty.pop(kind, set())
        if kind not in self.rows or timestamp - self.refreshed[kind] >= self.ttl:
            try:
                fresh = {row["id"]: row for row in self.store.all(kind, with_ids=True)}
            except Exception:
                with self.changes_lock:
                    self.dirty[kind].update(changed)
                raise
            previous = self.rows.get(kind, {})
            changed = set(previous) | set(fresh)
            self.refreshed[kind] = timestamp
            self.rows.setdefault(kind, {})
            self.by_robot.setdefault(kind, defaultdict(set))
            self.robots_by_id.setdefault(kind, {})
        else:
            fresh = {}
            try:
                for identifier in changed:
                    try:
                        fresh[identifier] = {
                            **self.store.get(kind, identifier),
                            "id": identifier,
                        }
                    except DomainError as exc:
                        if exc.code != 404:
                            raise
            except Exception:
                with self.changes_lock:
                    self.dirty[kind].update(changed)
                raise
        for identifier in changed:
            old, new = self.rows[kind].get(identifier), fresh.get(identifier)
            old_robot = self.robots_by_id[kind].get(identifier)
            new_robot = self.robot_resolver(new) if new is not None else None
            if old == new and old_robot == new_robot:
                continue
            if kind == "job" and old_robot != new_robot:
                with self.changes_lock:
                    for dependent in ("annotation", "check", "dataset_description"):
                        self.dirty[dependent].update(
                            row["id"]
                            for row in self.rows.get(dependent, {}).values()
                            if row.get("job_id", row["id"]) == identifier
                            and record_robot(row) == UNKNOWN_ROBOT
                        )
            if old is not None:
                self.by_robot[kind][old_robot].discard(identifier)
            if new is None:
                self.rows[kind].pop(identifier, None)
                self.robots_by_id[kind].pop(identifier, None)
            else:
                self.rows[kind][identifier] = new
                self.robots_by_id[kind][identifier] = new_robot
                self.by_robot[kind][new_robot].add(identifier)
            for robot in {old_robot, new_robot} - {None}:
                self.robot_versions[kind, robot] += 1
            self.tokens[kind, identifier] += 1
            self.versions[kind] += 1

    def select(self, kind, robot_type=None, *, omit=()):
        with self.lock:
            if kind in {"annotation", "check", "dataset_description"}:
                self._sync("job")
            self._sync(kind)
            rows = self.rows[kind]
            identifiers = (
                rows if robot_type is None else self.by_robot[kind].get(robot_type, ())
            )
            selected = [rows[identifier] for identifier in identifiers]
            selected.sort(
                key=lambda row: (row.get("created_at", 0), row["id"]), reverse=True
            )
            if omit:
                selected = [
                    {
                        **{
                            key: value
                            for key, value in row.items()
                            if key not in omit
                            or (key == "profile" and row.get("state") == "BLOCKED")
                        },
                        "robot_type": self.robot_resolver(row),
                    }
                    for row in selected
                ]
            return copy.deepcopy(selected)

    def version(self, kind, robot_type=None):
        with self.lock:
            self._sync(kind)
            return (
                self.versions[kind]
                if robot_type is None
                else self.robot_versions[kind, robot_type]
            )

    def lookup(self, kind, identifiers):
        with self.lock:
            if kind == "dataset_description":
                self._sync("job")
            self._sync(kind)
            return {
                identifier: copy.deepcopy(self.rows[kind][identifier])
                for identifier in identifiers
                if identifier in self.rows[kind]
            }

    def token(self, kind, identifier):
        with self.lock:
            return self.tokens[kind, identifier]

    def warm(self):
        for kind in INDEX_KINDS:
            self.select(kind)


class ReadCache:
    def __init__(self, ttl=30, capacity=4096):
        self.ttl, self.capacity = ttl, capacity
        self.lock = threading.RLock()
        self.entries = OrderedDict()
        self.pending = {}

    def get(self, key, load):
        with self.lock:
            timestamp = time.monotonic()
            found = self.entries.get(key)
            if found is not None and timestamp - found[0] < self.ttl:
                self.entries.move_to_end(key)
                return copy.deepcopy(found[1])
            pending = self.pending.get(key)
            owner = pending is None
            if owner:
                pending = Future()
                self.pending[key] = pending
        if not owner:
            return copy.deepcopy(pending.result())
        try:
            value = load()
            snapshot = copy.deepcopy(value)
            with self.lock:
                self.entries[key] = (time.monotonic(), snapshot)
                self.entries.move_to_end(key)
                while len(self.entries) > self.capacity:
                    self.entries.popitem(last=False)
            pending.set_result(snapshot)
            return value
        except BaseException as exc:
            pending.set_exception(exc)
            raise
        finally:
            with self.lock:
                self.pending.pop(key, None)


class WorkbenchLists:
    def __init__(
        self, settings, ingest, runner, datasets, annotations, upload_rows, job_response
    ):
        self.settings, self.ingest, self.runner = settings, ingest, runner
        self.datasets, self.annotations = datasets, annotations
        self.upload_rows, self.job_response = upload_rows, job_response
        self.index = RecordIndex(ingest.store, self._robot)
        self.cache = ReadCache()

    def _robot(self, row):
        robot = record_robot(row)
        job_id = row.get("job_id")
        if not job_id and "description" in row and "revision" in row:
            job_id = row["id"]
        if robot == UNKNOWN_ROBOT and job_id:
            job = self.index.rows.get("job", {}).get(job_id)
            if job:
                return record_robot(job)
        if robot != UNKNOWN_ROBOT or not row.get("manifest_hash"):
            return robot
        try:
            return self.ingest.manifest(row)["profile"]["robot_type"]
        except (DomainError, OSError, ValueError, KeyError):
            return UNKNOWN_ROBOT

    def metadata(self, job):
        key = ("metadata", job["id"], self.index.token("job", job["id"]), digest(job))

        def load():
            path = dataset_for_job(self.settings, job)
            info = json.loads((path / "meta/info.json").read_text(encoding="utf-8"))
            if not isinstance(info, dict):
                raise DomainError("数据集元数据必须为 JSON 对象", 409)
            return path, info

        return self.cache.get(key, load)

    def _project(self, kind, row, build, *, recycle=False):
        key = (kind, row["id"], digest(row))
        if recycle:
            key += (self.index.version("recycle"),)
        return self.cache.get(key, build)

    def _job(self, row):
        return self._project(
            "job",
            row,
            lambda: self.job_response(row, summary=True, metadata=self.metadata),
        )

    def robots(self):
        types = {runtime.robot_type for runtime in self.runner.catalog.all()}
        for kind in ("upload", "job"):
            types.update(
                self._robot(row)
                for row in self.index.select(
                    kind, omit=JOB_SUMMARY_OMIT if kind == "job" else ()
                )
                if row.get("state") != "DELETED"
            )
        return sorted(types)

    def page(self, resource, robot_type, *, offset=0, limit=50, q="", kind=None):
        if resource in {"collections", "datasets"}:
            uploads = self.index.select("upload", robot_type)
            uploads = [{**row, "robot_type": self._robot(row)} for row in uploads]
            if resource == "collections":
                uploads = visible_collection_uploads(uploads)
            groups = self.datasets.collections(uploads)
            for group in groups:
                group["created_at"] = max(
                    row.get("created_at", 0) for row in group["uploads"]
                )
        else:
            groups = []
        jobs = (
            self.index.select("job", robot_type, omit=JOB_SUMMARY_OMIT)
            if resource != "collections"
            else []
        )
        jobs = [
            job
            for job in jobs
            if job["state"] != "DELETED"
            and (resource == "datasets" or job["state"] not in DELETION_STATES)
        ]
        if resource == "collections":
            rows = groups
        elif resource in {"jobs", "merge-jobs", "results"}:
            rows = (
                [job for job in jobs if ready(job)]
                if resource == "results"
                else [
                    job
                    for job in jobs
                    if (job.get("operation") == "merge") == (resource == "merge-jobs")
                ]
            )
        elif resource in {
            "merge-candidates",
            "version-conversion-candidates",
            "annotation-candidates",
            "trim-candidates",
        }:
            rows = [job for job in jobs if ready(job)]
        elif resource == "annotations":
            rows = self.index.select("annotation", robot_type)
        elif resource == "datasets":
            rows = [dict(group, kind="raw") for group in groups]
            rows.extend(
                dict(job, kind="result")
                for job in jobs
                if ready(job) or job["state"] in DELETION_STATES
            )
            if kind:
                rows = [row for row in rows if row["kind"] == kind]
        else:
            raise DomainError("列表类型不存在", 404)
        if q:
            search = q.casefold()
            rows = [
                row
                for row in rows
                if search
                in f"{row.get('dataset_name', '')} {row.get('instruction', '')} {batch_name_for_job(row) or ''} {row.get('collection_date', '')} {row['id']}".casefold()
            ]
        rows.sort(key=lambda row: (row.get("created_at", 0), row["id"]), reverse=True)
        total = len(rows)
        selected = rows[offset : offset + limit]
        if resource == "collections":
            items = [self._collection(group) for group in selected]
        elif resource in {"jobs", "merge-jobs", "results"}:
            items = [self._job(row) for row in selected]
            if resource == "results":
                descriptions = self.index.lookup(
                    "dataset_description", [row["id"] for row in selected]
                )
                items = [
                    {
                        **item,
                        "dataset_description_preview": self._description_preview(
                            descriptions.get(item["id"], {}).get("description", "")
                        ),
                    }
                    for item in items
                ]
        elif resource in {"merge-candidates", "version-conversion-candidates"}:
            worker = (
                self.runner.merges
                if resource == "merge-candidates"
                else self.runner.version_conversions
            )
            items = [
                self._project(
                    resource,
                    row,
                    lambda row=row: worker.candidates(
                        jobs=[row], metadata=self.metadata
                    )[0],
                    recycle=True,
                )
                for row in selected
            ]
        elif resource == "trim-candidates":
            worker = getattr(self.runner, "trims", None)
            if worker is None:
                return {
                    "items": [],
                    "total": 0,
                    "offset": offset,
                    "limit": limit,
                    "available": False,
                }
            if "jobs" in inspect.signature(worker.candidates).parameters:
                items = [
                    self._project(
                        resource,
                        row,
                        lambda row=row: worker.candidates(
                            jobs=[row], metadata=self.metadata
                        )[0],
                        recycle=True,
                    )
                    for row in selected
                ]
            else:
                # Older deployments have a separate trim module. Keep it usable
                # until that module adopts the scoped candidate contract.
                candidates = self.cache.get(
                    (
                        "legacy-trim",
                        self.index.version("job"),
                        self.index.version("recycle"),
                    ),
                    worker.candidates,
                )
                by_id = {row["id"]: row for row in candidates}
                items = [by_id[row["id"]] for row in selected if row["id"] in by_id]
        elif resource == "annotation-candidates":
            annotation_rows = self.index.select("annotation", robot_type)
            items = self.annotations.candidates(
                jobs=selected, annotations=annotation_rows, metadata=self.metadata
            )
        elif resource == "datasets":
            items = [self._dataset(row) for row in selected]
        else:
            items = selected
        # Older dataset metadata can omit robot_type. Every source row above
        # was already selected through the authoritative robot index.
        items = [
            dict(item, robot_type=item.get("robot_type") or robot_type)
            for item in items
        ]
        response = {"items": items, "total": total, "offset": offset, "limit": limit}
        if resource == "datasets":
            fields = (
                "id",
                "state",
                "created_at",
                "finished_at",
                "dataset_name",
                "robot_type",
                "kind",
                "error",
            )
            response["deletions"] = [
                {field: op.get(field) for field in fields}
                for op in self.index.select("dataset_deletion", robot_type)[:50]
            ]
        return response

    @staticmethod
    def _description_preview(text):
        return text if len(text) <= 300 else text[:300].rstrip() + "…"

    def _collection(self, group, *, detail=False):
        rows = group["uploads"]
        revision = digest(
            [(row["id"], self.index.token("upload", row["id"])) for row in rows]
        )
        if detail:
            uploads = [
                self._project("upload", row, lambda row=row: self.upload_rows([row])[0])
                for row in rows
            ]
        else:
            uploads = [
                {key: value for key, value in row.items() if key in UPLOAD_LIST_FIELDS}
                for row in rows
            ]
        return {
            **group,
            "uploads": uploads,
            "revision": revision,
            "conversion_checks_pending": not detail,
        }

    def collection_detail(self, robot_type, identifier):
        uploads = [
            {**row, "robot_type": self._robot(row)}
            for row in self.index.select("upload", robot_type)
            if row["state"] == "COMMITTED"
        ]
        for group in self.datasets.collections(uploads):
            if group["id"] == identifier:
                return self._collection(group, detail=True)
        raise DomainError("数据集不存在或不属于所选机器人", 404)

    def _dataset(self, row):
        raw = row["kind"] == "raw"
        records = row["uploads"] if raw else [row]
        first = records[0]
        path = first.get("raw_path") if raw else row.get("dataset_path")
        batch_name = None if raw else batch_name_for_job(row)
        states = {record["state"] for record in records}
        state = (
            "DELETE_FAILED"
            if "DELETE_FAILED" in states
            else "DELETING"
            if "DELETING" in states
            else "READY"
            if states == {"COMMITTED"} or not raw
            else "UPLOADING"
        )
        return {
            "id": row["id"],
            "kind": row["kind"],
            "dataset_name": row.get("dataset_name")
            or row.get("instruction")
            or row["id"],
            "robot_type": self._robot(first),
            "state": state,
            "episode_count": row.get("episode_count") or row.get("source_count"),
            "total_bytes": row.get("total_bytes"),
            "target_format": row.get("target_format"),
            "paths": [path] if path else [],
            "deletable": state == "READY" and not row.get("immutable_delivery"),
            "validation_pending": True,
            "reason": "外部交付数据集为只读" if row.get("immutable_delivery") else "",
            **({"batch_name": batch_name} if batch_name else {}),
        }

    def deletion_preview(self, kind, identifier):
        with self.runner.lock, self.ingest.lock:
            name, robot, records, paths = self.datasets._selection(kind, identifier)
            paths = self.datasets._validate(records, paths)
            return {
                "id": identifier,
                "kind": kind,
                "dataset_name": name,
                "robot_type": robot,
                "paths": [str(p) for p in paths],
            }

    def status(self, robot_type):
        jobs = self.index.select("job", robot_type, omit=JOB_SUMMARY_OMIT)
        uploads = self.index.select("upload", robot_type)
        visible_uploads = visible_collection_uploads(uploads)
        annotations = self.index.select("annotation", robot_type)
        checks = self.index.select("check", robot_type)
        fields = (
            "id",
            "state",
            "stage",
            "progress",
            "error",
            "started_at",
            "finished_at",
            "dataset_name",
            "robot_type",
        )
        return {
            "revisions": {
                kind: self.index.version(
                    kind, None if kind == "recycle" else robot_type
                )
                for kind in INDEX_KINDS
            },
            "datasets_revision": digest(
                [
                    (
                        row["id"],
                        row["state"],
                        row.get("dataset_path"),
                        row.get("dataset_ready"),
                    )
                    for row in jobs
                ]
            ),
            "active_jobs": [
                {key: row.get(key) for key in fields}
                for row in jobs
                if row["state"] in ACTIVE
            ],
            "active_annotations": [
                row["id"] for row in annotations if row["state"] in ACTIVE
            ],
            "active_checks": [row["id"] for row in checks if row["state"] in ACTIVE],
            "uploading": sum(
                row["state"] in {"UPLOADING", "COMMITTING"} for row in uploads
            ),
            "counts": {
                "uploads": len(visible_uploads),
                "committed": len(visible_uploads),
                "collections": len(
                    {
                        row.get("collection_id") or "unavailable-" + row["id"]
                        for row in visible_uploads
                    }
                ),
                "jobs": sum(
                    row["state"] not in DELETION_STATES
                    and row.get("operation") != "merge"
                    for row in jobs
                ),
                "results": sum(ready(row) for row in jobs),
                "datasets": len(
                    {
                        row.get("collection_id") or "unavailable-" + row["id"]
                        for row in visible_uploads
                    }
                )
                + sum(ready(row) for row in jobs),
            },
        }
