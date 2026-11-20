"""Persistent, bounded, read-only scoring for registered Viewer datasets."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pipeline.shared.core import DomainError, executable_path
from pipeline.shared.identity import public_execution_record

SCORING_VERSION = "doctor-ff947533-vector-shards-v1"


def dataset_signature(root: Path) -> str:
    paths = [root / "meta/info.json", *sorted((root / "data").rglob("*.parquet"))]
    entries = []
    for path in paths:
        if any(p.is_symlink() for p in [path, *path.parents]):
            raise DomainError("评分输入不允许包含符号链接", 400)
        stat = path.stat()
        entries.append(
            (str(path.relative_to(root)), stat.st_ino, stat.st_size, stat.st_mtime_ns)
        )
    return hashlib.sha256(json.dumps([SCORING_VERSION, entries]).encode()).hexdigest()


class EpisodeScoreCache:
    def __init__(self, settings, store, runner=None):
        self.settings, self.store = settings, store
        self.runner = runner
        self.pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="episode-score"
        )
        self.lock = threading.RLock()
        self.pending = {}
        self.claims = {}
        self.stopping = threading.Event()

    def close(self):
        self.stopping.set()
        self.pool.shutdown(wait=True, cancel_futures=True)
        with self.lock:
            claims = self.claims
            self.claims = {}
        for claim in claims.values():
            self._release_claim(claim)

    def _acquire_claim(self, key):
        directory = self.store.path / "episode-score-locks"
        directory.mkdir(exist_ok=True)
        claim = (directory / f"{key}.lock").open("a")
        try:
            fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            claim.close()
            return None
        return claim

    @staticmethod
    def _release_claim(claim):
        if claim is None or claim.closed:
            return
        try:
            fcntl.flock(claim, fcntl.LOCK_UN)
        finally:
            claim.close()

    def _claim_is_held(self, key):
        claim = self._acquire_claim(key)
        if claim is None:
            return True
        self._release_claim(claim)
        return False

    def _persisted(self, key, signature):
        try:
            result = self.store.get("episode-score", key)
        except DomainError as exc:
            if exc.code != 404:
                raise
            return {"state": "NONE"}
        if result.get("signature") != signature:
            return {"state": "NONE"}
        if not isinstance(result.get("state"), str):
            return {"state": "NONE"}
        return result or {"state": "NONE"}

    def _identity(self, preview, principal=None):
        root = Path(preview["dataset_path"])
        submitter = (
            principal.username
            if (
                principal is not None
                and not principal.is_admin
                and principal.auth_method == "h800-ssh"
            )
            else "service-account"
        )
        key = hashlib.sha256(
            json.dumps(
                {"path": str(root), "submitted_as": submitter},
                sort_keys=True,
            ).encode()
        ).hexdigest()
        return root, key, dataset_signature(root)

    def status(self, preview, *, principal=None):
        with self.lock:
            _, key, signature = self._identity(preview, principal)
            result = self._persisted(key, signature)
            if result["state"] not in {"QUEUED", "RUNNING"}:
                return public_execution_record(result)
            pending = self.pending.get(key)
            if pending is None or pending.done():
                if not self._claim_is_held(key):
                    return {"state": "NONE"}
            return public_execution_record(result)

    def start(self, preview, *, principal=None):
        with self.lock:
            root, key, signature = self._identity(preview, principal)
            current = self.status(preview, principal=principal)
            if current["state"] in {"QUEUED", "RUNNING", "COMPLETED"}:
                return current
            claim = self._acquire_claim(key)
            if claim is None:
                current = self._persisted(key, signature)
                if current["state"] != "NONE":
                    return public_execution_record(current)
                return {"state": "RUNNING", "signature": signature}

            try:
                current = self._persisted(key, signature)
                if current["state"] == "COMPLETED":
                    self._release_claim(claim)
                    return public_execution_record(current)
                tool = executable_path(self.settings.letools_bin)
                python = Path(tool).with_name("python") if tool else None
                if (
                    python is None
                    or not python.is_file()
                    or not os.access(python, os.X_OK)
                ):
                    raise DomainError(
                        "未找到 LeTools Python，无法运行 Episode 评分", 503
                    )
                if self.runner is not None:
                    self.runner.require_principal_path_access(principal, root)
                    metadata = self.runner.submission_metadata(principal)
                elif principal is not None:
                    metadata = principal.job_metadata()
                else:
                    metadata = {
                        "owner_user_id": None,
                        "owner_uid": None,
                        "submitted_as": "service-account",
                        "slurm_user": None,
                        "slurm_execution_id": None,
                        "submission_auth": "internal",
                    }
                result = {
                    "state": "QUEUED",
                    "signature": signature,
                    "created_at": time.time(),
                    **metadata,
                }
                self.store.put("episode-score", key, result)
                self.claims[key] = claim
            except BaseException:
                if self.claims.get(key) is claim:
                    self.claims.pop(key, None)
                self._release_claim(claim)
                raise
            try:
                pending = self.pool.submit(
                    self._execute_claimed,
                    root,
                    key,
                    signature,
                    str(python),
                    claim,
                    metadata,
                )
            except BaseException:
                self.claims.pop(key, None)
                self._release_claim(claim)
                raise
            self.pending[key] = pending
            return public_execution_record(result)

    def _execute_claimed(self, root, key, signature, python, claim, metadata):
        try:
            self._run(root, key, signature, python, metadata)
        finally:
            with self.lock:
                if self.claims.get(key) is claim:
                    self.claims.pop(key, None)
                self._release_claim(claim)

    def _run(self, root, key, signature, python, metadata):
        result = {
            "state": "RUNNING",
            "signature": signature,
            "created_at": time.time(),
            **metadata,
        }
        self.store.put("episode-score", key, result)
        try:
            worker = Path(__file__).with_name("episode_score_worker.py")
            argv = [python, str(worker), str(root)]
            if self.runner is None:
                completed = subprocess.run(
                    argv,
                    capture_output=True,
                    text=True,
                    timeout=self.settings.command_timeout,
                    env=self.settings.tool_environment(),
                )
            else:
                with self.runner.bind_record_session(metadata):
                    completed = self.runner.run_captured(
                        argv,
                        timeout=self.settings.command_timeout,
                    )
            if completed.returncode:
                raise ValueError(
                    completed.stderr[-1200:] or "评分程序执行失败"
                )
            report = json.loads(completed.stdout)
            json.dumps(report, allow_nan=False)
            rows = report["episodes"]
            total = report["total_episodes"]
            if len(rows) != total or [r["episode_index"] for r in rows] != list(
                range(total)
            ):
                raise ValueError("评分结果未覆盖全部 Episode")
            if dataset_signature(root) != signature:
                raise ValueError("评分期间数据集发生变化，请重新计算")
            result.update(state="COMPLETED", report=report)
        except Exception as exc:
            result.update(state="FAILED", error=str(exc))
        result["finished_at"] = time.time()
        self.store.put("episode-score", key, result)
