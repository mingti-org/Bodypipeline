from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from pipeline.shared.core import DomainError, Store, executable_path
from pipeline.shared.security import install_security

from .datasets import dataset_for_job
from .instruction_review import (
    ApplyInstructionsRequest,
    EpisodeSaveRequest,
    TemplateSaveRequest,
)
from .subtask_review import DraftRequest, RevisionExportRequest
from .task_annotations import TemplateRequest
from .viewer_data import PreviewRegistry
from .viewer_gateway import (
    SERVICE_KEY,
    SERVICE_KIND,
    ManualTrimRequest,
    RecycleSelection,
    ViewerTransport,
)


class ViewerManager:
    """Keeps one read-only Visualizer running and registers datasets for it.

    The Visualizer application is dataset-agnostic: it fetches
    ``<origin>/<org>/<dataset>/resolve/main/...``. The gateway serves that URL
    shape for every registered dataset, so opening a preview only records a
    mapping. No process is started or stopped per dataset, which removes the
    single-session conflict and the port races that came with it.
    """

    def __init__(self, settings, store):
        self.settings, self.store = settings, store
        self.process = None
        self.log = None
        self.lock = threading.RLock()
        self.registry = PreviewRegistry(store, settings.viewer_ttl)

    # ------------------------------------------------------------ lifecycle
    def _ports_busy(self):
        for port in (self.settings.viewer_ui_port,):
            with socket.socket() as sock:
                try:
                    sock.bind(("127.0.0.1", port))
                except OSError:
                    return True
        return False

    def _service(self):
        try:
            return self.store.get(SERVICE_KIND, SERVICE_KEY)
        except DomainError as exc:
            if exc.code == 404:
                return None
            raise

    def _put_service(self, **fields):
        row = self._service() or {}
        row.update(fields)
        self.store.put(SERVICE_KIND, SERVICE_KEY, row)
        return row

    def _kill_group(self, pid):
        if not isinstance(pid, int) or pid <= 1:
            return
        try:
            os.killpg(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                os.killpg(pid, 0)
            except (ProcessLookupError, PermissionError):
                return
            time.sleep(0.1)
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    def _reap_matching_processes(self):
        """Kill our own Visualizer processes that predate PID tracking.

        Only processes whose command line names this server's viewer cache
        directory are touched; anything else on the port is left alone.
        """

        marker = str(self.settings.root / "viewer-cache").encode()
        proc = Path("/proc")
        if not proc.is_dir():
            return
        for entry in proc.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ")
            except OSError:
                continue
            pid = int(entry.name)
            if marker not in cmdline or pid <= 1 or pid == os.getpid():
                continue
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                continue
        deadline = time.time() + 10
        while time.time() < deadline and self._ports_busy():
            time.sleep(0.1)

    def _clear_stale(self):
        if self.process is not None or not self._ports_busy():
            return
        service = self._service() or {}
        if service.get("pid"):
            self._kill_group(service["pid"])
        if self._ports_busy():
            self._reap_matching_processes()

    def recover(self):
        """Reap any Visualizer a previous server process left behind."""

        service = self._service()
        if service is None:
            return
        if service.get("state") in {"STARTING", "READY"}:
            self._kill_group(service.get("pid"))
            if self._ports_busy():
                self._reap_matching_processes()
            self._put_service(
                state="INTERRUPTED",
                error="Server 重启，已清理上一次的预览进程",
                pid=None,
            )

    def ensure_service(self):
        """Start the Visualizer if it is not already running."""

        with self.lock:
            if self.process and self.process.poll() is None:
                return self._service()
            executable = executable_path(self.settings.letools_bin)
            if not executable:
                raise DomainError("未安装 letools / Visualizer 依赖")
            self._clear_stale()
            if self._ports_busy():
                raise DomainError(
                    f"预览内部端口 {self.settings.viewer_ui_port} 已被其他程序占用，不自动终止其他进程"
                )
            log_path = self.settings.root / "reports/viewer.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log = log_path.open("ab")
            env = self.settings.tool_environment()
            env["NEXT_TELEMETRY_DISABLED"] = "1"
            wrapper = [
                sys.executable,
                "-m",
                "pipeline.server.viewer_bootstrap",
                "--executable",
                executable,
                "--cache",
                str(self.settings.root / "viewer-cache"),
                "--data-origin",
                self.settings.viewer_origin.rstrip("/") + "/_data",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.settings.viewer_ui_port),
            ]
            if self.settings.viewer_bun:
                wrapper += ["--bun", str(Path(self.settings.viewer_bun).resolve())]
            self.process = subprocess.Popen(
                wrapper,
                stdout=self.log,
                stderr=self.log,
                start_new_session=True,
                env=env,
            )
            return self._put_service(
                state="STARTING",
                pid=self.process.pid,
                log_path=str(log_path),
                started_at=time.time(),
                error=None,
            )

    def autostart(self):
        """Bring the always-on Visualizer up; record why if it cannot start."""

        with self.lock:
            try:
                self.ensure_service()
            except DomainError as exc:
                self._put_service(state="UNAVAILABLE", pid=None, error=exc.message)
            except (OSError, subprocess.SubprocessError) as exc:
                self._put_service(state="FAILED", pid=None, error=str(exc))

    def poll(self):
        """Mark the service READY, restart it if it died, and prune old links."""

        with self.lock:
            self.registry.prune()
            service = self._service()
            if self.process is None:
                if (
                    service
                    and service.get("state") == "FAILED"
                    and time.time() >= service.get("retry_at", 0)
                ):
                    self.autostart()
                return
            if self.process.poll() is not None:
                code = self.process.returncode
                self.process = None
                if self.log:
                    self.log.close()
                    self.log = None
                self._put_service(
                    state="FAILED",
                    pid=None,
                    retry_at=time.time() + 30,
                    error=f"Visualizer 退出：{code}；检查 reports/viewer.log",
                )
                return
            if service and service.get("state") == "STARTING":
                try:
                    response = httpx.get(
                        f"http://127.0.0.1:{self.settings.viewer_ui_port}/",
                        timeout=1,
                        trust_env=False,
                    )
                    if response.status_code < 500:
                        self._put_service(state="READY")
                except httpx.HTTPError:
                    pass

    def stop(self):
        """Stop the Visualizer. Never depends on readable state, so shutdown is safe."""

        with self.lock:
            if self.process and self.process.poll() is None:
                self._kill_group(self.process.pid)
                try:
                    self.process.wait(5)
                except subprocess.TimeoutExpired:
                    pass
            self.process = None
            deadline = time.time() + 10
            while time.time() < deadline and self._ports_busy():
                time.sleep(0.1)
            if self.log:
                self.log.close()
                self.log = None
            try:
                return self._put_service(state="STOPPED", pid=None)
            except DomainError:
                # A corrupt state file must not abort shutdown or hold the lock.
                return {"state": "STOPPED"}

    # --------------------------------------------------------------- opening
    def open(self, job):
        """Register ``job``'s dataset and return the link that shows it."""

        data = dataset_for_job(self.settings, job)
        service = self.ensure_service()
        row = self.registry.register(job["id"], data)
        return self._describe(row, service)

    def close(self, job_id):
        self.registry.remove(job_id)
        return {"state": "CLOSED", "job_id": job_id}

    def _describe(self, row, service):
        origin = self.settings.viewer_origin.rstrip("/")
        return {
            "state": (service or {}).get("state", "STARTING"),
            "job_id": row["job_id"],
            "url": origin + row["page_path"],
            "expires_at": row["expires_at"],
        }

    def status(self):
        service = self._service() or {"state": "STOPPED"}
        return {
            "state": service.get("state", "STOPPED"),
            "error": service.get("error"),
            "previews": [
                self._describe(row, service) for row in self.registry.active()
            ],
        }


def create_gateway(settings):
    settings.role = "viewer"
    store = Store(settings.root / "state", legacy_name="server.sqlite")
    registry = PreviewRegistry(store, settings.viewer_ttl)
    transport = ViewerTransport(settings, store, registry)

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            await asyncio.to_thread(transport.close)

    app = FastAPI(title="LeRobot Viewer Gateway", lifespan=lifespan)
    app.state.viewer_transport = transport
    app.state.episode_scores = transport.scores
    # The gateway is reachable beyond loopback once it is bound to a routable
    # address, and it serves raw dataset files, so it shares the Server's token.
    install_security(app, settings)

    @app.exception_handler(DomainError)
    async def domain_error(_request, exc):
        return JSONResponse({"detail": exc.message}, status_code=exc.code)

    transport.register(app, embedded=False)

    def score_preview(dataset_id):
        row = transport.registry.lookup(f"bp/{dataset_id}")
        if row is None:
            raise DomainError("预览已过期或不存在，请在管理台重新打开", 404)
        return row

    async def review_proxy(dataset_id, suffix, body=None):
        # Only server-owned preview identities are accepted, never a local_path
        # or arbitrary proxy destination. The Server alone writes draft state.
        score_preview(dataset_id)
        return await annotation_proxy(f"/api/subtask-review/{dataset_id}/{suffix}", body)

    async def annotation_proxy(route, body=None):
        url = settings.server_url.rstrip("/") + route
        try:
            async with httpx.AsyncClient(trust_env=False) as client:
                response = await client.request(
                    "GET" if body is None else "POST", url,
                    json=body.model_dump() if body is not None else None,
                    headers={"Authorization": "Bearer " + settings.token},
                    timeout=settings.command_timeout,
                )
        except httpx.HTTPError as exc:
            raise DomainError("Server 服务暂不可用，请稍后重试", 503) from exc
        try:
            payload = response.json()
        except ValueError as exc:
            raise DomainError("Server 服务返回无效响应", 502) from exc
        return JSONResponse(payload, status_code=response.status_code, headers={"Cache-Control": "no-store"})

    async def instruction_proxy(dataset_id, suffix, body=None, method="GET"):
        score_preview(dataset_id)
        suffix = f"/{suffix}" if suffix else ""
        url = settings.server_url.rstrip("/") + f"/api/instructions/{dataset_id}{suffix}"
        try:
            async with httpx.AsyncClient(trust_env=False) as client:
                response = await client.request(
                    method,
                    url,
                    json=body.model_dump() if body is not None else None,
                    headers={"Authorization": "Bearer " + settings.token},
                    timeout=settings.command_timeout,
                )
        except httpx.HTTPError as exc:
            raise DomainError(
                "Server 语言指令服务暂不可用，请稍后重试", 503
            ) from exc
        try:
            payload = response.json()
        except ValueError as exc:
            raise DomainError("Server 语言指令服务返回无效响应", 502) from exc
        return JSONResponse(
            payload,
            status_code=response.status_code,
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/task-templates")
    async def task_templates():
        return await annotation_proxy("/api/task-templates")

    @app.post("/api/task-templates")
    async def save_task_templates(body: TemplateRequest):
        return await annotation_proxy("/api/task-templates", body)

    @app.get("/api/instructions/{dataset_id}")
    async def instruction_dataset(dataset_id: str):
        return await instruction_proxy(dataset_id, "")

    @app.put("/api/instructions/{dataset_id}/template")
    async def instruction_template_save(
        dataset_id: str, body: TemplateSaveRequest
    ):
        return await instruction_proxy(
            dataset_id, "template", body, method="PUT"
        )

    @app.get("/api/instructions/{dataset_id}/episodes/{episode}")
    async def instruction_episode(dataset_id: str, episode: int):
        return await instruction_proxy(dataset_id, f"episodes/{episode}")

    @app.put("/api/instructions/{dataset_id}/episodes/{episode}")
    async def instruction_episode_save(
        dataset_id: str, episode: int, body: EpisodeSaveRequest
    ):
        return await instruction_proxy(
            dataset_id, f"episodes/{episode}", body, method="PUT"
        )

    @app.post("/api/instructions/{dataset_id}/episodes/{episode}/apply")
    async def instruction_episode_apply(
        dataset_id: str, episode: int, body: ApplyInstructionsRequest
    ):
        return await instruction_proxy(
            dataset_id,
            f"episodes/{episode}/apply",
            body,
            method="POST",
        )

    @app.post("/api/instructions/{dataset_id}/apply")
    async def instruction_apply(
        dataset_id: str, body: ApplyInstructionsRequest
    ):
        return await instruction_proxy(dataset_id, "apply", body, method="POST")

    @app.get("/api/instructions/{dataset_id}/apply/{apply_id}")
    async def instruction_apply_status(dataset_id: str, apply_id: str):
        if not apply_id or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for character in apply_id
        ):
            raise DomainError("语言指令任务编号无效", 404)
        return await instruction_proxy(dataset_id, f"apply/{apply_id}")

    @app.get("/api/subtask-review/{dataset_id}/episodes/{episode}")
    async def review_episode(dataset_id: str, episode: int):
        return await review_proxy(dataset_id, f"episodes/{episode}")

    @app.post("/api/subtask-review/{dataset_id}/episodes/{episode}")
    async def review_save(dataset_id: str, episode: int, body: DraftRequest):
        return await review_proxy(dataset_id, f"episodes/{episode}", body)

    @app.post("/api/subtask-review/{dataset_id}/exports")
    async def review_export(dataset_id: str, body: RevisionExportRequest):
        return await review_proxy(dataset_id, "exports", body)

    @app.get("/api/subtask-review/{dataset_id}/exports/{export_id}")
    async def review_export_status(dataset_id: str, export_id: str):
        if not export_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in export_id):
            raise DomainError("修订导出任务编号无效", 404)
        return await review_proxy(dataset_id, f"exports/{export_id}")

    @app.get("/api/episode-scores/{dataset_id}")
    async def score_status(dataset_id: str):
        row = score_preview(dataset_id)
        result = await asyncio.to_thread(transport.scores.status, row)
        return JSONResponse(result, headers={"Cache-Control": "no-store"})

    @app.post("/api/episode-scores/{dataset_id}")
    async def score_start(dataset_id: str):
        row = score_preview(dataset_id)
        result = await asyncio.to_thread(transport.scores.start, row)
        return JSONResponse(
            result, status_code=202, headers={"Cache-Control": "no-store"}
        )

    @app.post("/api/manual-trims/{dataset_id}", status_code=202)
    async def manual_trim_start(dataset_id: str, body: ManualTrimRequest):
        # The source is always the registered preview's own job, so the viewer
        # can never trim a dataset the caller does not have a preview for.
        row = score_preview(dataset_id)
        return await annotation_proxy(
            f"/api/jobs/{row['job_id']}/manual-trims", body
        )

    async def manual_job(dataset_id: str, job_id: str):
        """Fetch a trim job only when it belongs to this preview's dataset."""

        row = score_preview(dataset_id)
        if not job_id or any(
            char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for char in job_id
        ):
            raise DomainError("裁剪任务编号无效", 404)
        url = settings.server_url.rstrip("/") + f"/api/jobs/{job_id}"
        try:
            async with httpx.AsyncClient(trust_env=False) as client:
                response = await client.get(
                    url,
                    headers={"Authorization": "Bearer " + settings.token},
                    timeout=settings.command_timeout,
                )
        except httpx.HTTPError as exc:
            raise DomainError("Server 裁剪服务暂不可用，请稍后重试", 503) from exc
        try:
            payload = response.json()
        except ValueError as exc:
            raise DomainError("Server 裁剪服务返回无效响应", 502) from exc
        if response.status_code == 200 and (
            not isinstance(payload, dict)
            or payload.get("operation") != "manual_trim"
            or payload.get("source_job_id") != row["job_id"]
        ):
            raise DomainError("裁剪任务不存在", 404)
        return payload, response.status_code

    @app.get("/api/manual-trims/{dataset_id}/jobs/{job_id}")
    async def manual_trim_status(dataset_id: str, job_id: str):
        payload, status = await manual_job(dataset_id, job_id)
        return JSONResponse(
            payload, status_code=status, headers={"Cache-Control": "no-store"}
        )

    @app.post("/api/manual-trims/{dataset_id}/jobs/{job_id}/preview")
    async def manual_trim_preview(dataset_id: str, job_id: str):
        payload, _status = await manual_job(dataset_id, job_id)
        if payload.get("state") not in {
            "COMPLETED",
            "CONVERTED",
            "SYNTHETIC_READY",
            "QUARANTINED",
        }:
            raise DomainError("裁剪尚未完成，暂时无法预览", 409)
        url = settings.server_url.rstrip("/") + f"/api/jobs/{job_id}/viewer"
        try:
            async with httpx.AsyncClient(trust_env=False) as client:
                response = await client.post(
                    url,
                    headers={"Authorization": "Bearer " + settings.token},
                    timeout=settings.command_timeout,
                )
        except httpx.HTTPError as exc:
            raise DomainError("Server 预览服务暂不可用，请稍后重试", 503) from exc
        try:
            opened = response.json()
        except ValueError as exc:
            raise DomainError("Server 预览服务返回无效响应", 502) from exc
        if response.status_code >= 400:
            detail = (
                opened.get("detail")
                if isinstance(opened, dict)
                else "无法打开裁剪结果"
            )
            raise DomainError(str(detail), response.status_code)
        return JSONResponse(
            {"url": opened.get("url")}, headers={"Cache-Control": "no-store"}
        )

    @app.get("/api/recycle/{dataset_id}")
    async def recycle_status(dataset_id: str):
        result = await asyncio.to_thread(
            transport.recycle.status, f"bp/{dataset_id}"
        )
        return JSONResponse(result, headers={"Cache-Control": "no-store"})

    @app.post("/api/recycle/{dataset_id}/{action}")
    async def recycle_change(dataset_id: str, action: str, selection: RecycleSelection):
        if action not in {"trash", "restore", "purge"}:
            raise DomainError("不支持的回收站操作", 404)
        result = await asyncio.to_thread(
            transport.recycle.change,
            f"bp/{dataset_id}",
            action,
            sorted(set(selection.episodes)),
            selection.revision,
        )
        return JSONResponse(result, headers={"Cache-Control": "no-store"})

    return app


__all__ = ["ViewerManager", "create_gateway"]
