from __future__ import annotations

import asyncio
import html
import json
import subprocess
import threading
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt

from pipeline.shared.core import DomainError, executable_path
from pipeline.shared.web import ASSETS, themed_html

from .episode_scores import EpisodeScoreCache
from .recycle import RecycleBin
from .viewer_data import serve_dataset_file

VIEWER_PROXY_PREFIXES: tuple[str, ...] = ("/bp/", "/_next/")
VIEWER_ASSET_PATHS: tuple[str, ...] = (
    "/favicon.ico",
    "/bodypipeline-icon.png",
    "/bodypipeline-mark.png",
)
FORWARDED_UI_HEADERS = {
    "range",
    "if-range",
    "accept",
    "accept-encoding",
    "if-none-match",
    "rsc",
    "next-router-state-tree",
    "next-router-prefetch",
    "next-url",
}
RESPONSE_UI_HEADERS = {
    "content-type",
    "content-length",
    "content-range",
    "accept-ranges",
    "content-encoding",
    "etag",
    "vary",
    "location",
    "cache-control",
}

SERVICE_KEY = "service"
SERVICE_KIND = "viewer"

STARTING_PAGE = themed_html("""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="2">
<title>正在启动数据预览</title></head>
<body style="margin:0;background:var(--page-bg,#f7f8fa);color:var(--page-ink,#292e36);font:15px system-ui,'PingFang SC','Microsoft YaHei',sans-serif">
<main class="viewer-starting-card" style="max-width:620px;margin:15vh auto;padding:32px;background:#fff;border:1px solid #e0e3e8;border-radius:8px">
<h1 style="font-size:22px;margin:0 0 12px">正在启动 LeRobot Visualizer</h1>
<p style="line-height:1.7;margin:0;color:#6b7280">首次启动可能需要安装或准备前端依赖。页面每 2 秒自动刷新，就绪后会直接显示数据集。</p>
</main></body></html>""")


def embedded_enabled(settings) -> bool:
    config = getattr(settings, "viewer_gateway", {})
    return isinstance(config, dict) and config.get("embedded") is True


def _gateway_state(store):
    try:
        return store.get(SERVICE_KIND, SERVICE_KEY)
    except DomainError as exc:
        if exc.code == 404:
            return None
        raise


def _doctor_html(payload: dict, dataset_path: str, sample: str) -> str:
    """Render the Doctor payload with the shared workbench theme."""

    severity = str(payload.get("overall_severity", "PASS"))
    sections = []
    for check in payload.get("checks", []):
        messages = "".join(
            "<li><span class='message-severity'>"
            + html.escape(str(m.get("severity", "")))
            + "</span> "
            + html.escape(str(m.get("message", "")))
            + "</li>"
            for m in check.get("messages", [])
        )
        sections.append(
            "<section class='doctor-check'><div class='check-head'><h2>"
            + html.escape(str(check.get("name", "Check")))
            + "</h2><span class='severity "
            + html.escape(str(check.get("severity", "PASS")).lower())
            + "'>"
            + html.escape(str(check.get("severity", "PASS")))
            + "</span></div><ul>"
            + messages
            + "</ul></section>"
        )
    return themed_html(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport"
content="width=device-width,initial-scale=1"><title>LeRobot Doctor</title>
<style>
:root {{ color-scheme: light; }}
body {{ margin: 0; background: #f7f8fa; color: #292e36;
font-family: Inter, ui-sans-serif, system-ui, "PingFang SC", "Microsoft YaHei", sans-serif; }}
header {{ padding: 20px 24px 16px; border-bottom: 1px solid #e8ebf1; background: #fff; }}
h1 {{ margin: 0 0 6px; font-size: 20px; }}
.meta {{ font-size: 13px; color: #6b7280; }}
main {{ padding: 16px 24px 32px; display: grid; gap: 10px; }}
section {{ border: 1px solid #e3e7ee; border-radius: 6px; background: #fff; padding: 12px 14px; }}
.check-head {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; }}
h2 {{ margin: 0; font-size: 14px; color: #3b434f; }}
.severity {{ border: 1px solid currentColor; border-radius: 4px; padding: 2px 7px;
font: 700 11px ui-monospace, monospace; }}
.pass {{ color: #258467; }} .warn {{ color: #9a741b; }} .fail {{ color: #c04b59; }}
ul {{ margin: 9px 0 0; padding-left: 19px; font-size: 12px; line-height: 1.55; color: #4f5866; }}
.message-severity {{ color: #8a919d; font: 700 10px ui-monospace, monospace; }}
</style></head><body><header class="doctor-header"><h1>LeRobot Doctor
<span class="severity {html.escape(severity.lower())}">{html.escape(severity)}</span></h1>
<div class="meta">{html.escape(dataset_path)} &middot; {html.escape(sample)} &middot;
{html.escape(str(payload.get("total_episodes", "?")))} episodes &middot;
{html.escape(str(payload.get("total_frames", "?")))} frames</div></header>
<main>{"".join(sections)}</main></body></html>""")


class DoctorCache:
    """Computes Doctor once per dataset and reuses it until the link expires."""

    def __init__(self, settings):
        self.settings = settings
        self.lock = threading.Lock()
        self.cache: dict[str, tuple[float, dict]] = {}

    def payload(self, row: dict) -> dict:
        key = row["dataset_path"]
        stamp = (Path(key) / "meta/info.json").stat().st_mtime
        with self.lock:
            cached = self.cache.get(key)
            if cached and cached[0] == stamp:
                return cached[1]
        tool = executable_path(self.settings.letools_bin)
        if not tool:
            raise DomainError("未安装 letools / Doctor 依赖", 503)
        result = subprocess.run(
            [tool, "doctor", "check", key, "--json", "--max-episodes", "20"],
            capture_output=True,
            text=True,
            timeout=self.settings.command_timeout,
            env=self.settings.tool_environment(),
        )
        # lerobot-doctor exits 1 on a FAIL severity but still prints a report.
        if result.returncode not in (0, 1):
            raise DomainError("Doctor 运行失败：" + result.stderr[-400:], 502)
        try:
            payload = json.loads(result.stdout)
        except ValueError as exc:
            raise DomainError("Doctor 未返回有效报告", 502) from exc
        with self.lock:
            self.cache[key] = (stamp, payload)
        return payload


class ManualTrimRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    episode_index: StrictInt = Field(ge=0)
    remove_ranges: list[tuple[StrictInt, StrictInt]] = Field(
        min_length=1, max_length=100
    )
    name: str | None = Field(default=None, min_length=1, max_length=2000)
    run_doctor: StrictBool = False
    in_place: StrictBool = True


class RecycleSelection(BaseModel):
    episodes: list[StrictInt] = Field(min_length=1, max_length=100000)
    revision: StrictInt = Field(ge=0)


async def proxy_next_response(request: Request, url: str):
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() in FORWARDED_UI_HEADERS
    }
    client = httpx.AsyncClient(timeout=120, trust_env=False)
    try:
        response = await client.send(
            client.build_request(request.method, url, headers=headers),
            stream=True,
            follow_redirects=False,
        )
    except Exception:
        await client.aclose()
        raise DomainError("预览后端不可达", 502)

    async def chunks():
        try:
            async for data in response.aiter_raw():
                yield data
        finally:
            await response.aclose()
            await client.aclose()

    outgoing = {
        k: v
        for k, v in response.headers.items()
        if k.lower() in RESPONSE_UI_HEADERS
    }
    return StreamingResponse(
        chunks(), status_code=response.status_code, headers=outgoing
    )


class ViewerTransport:
    def __init__(self, settings, store, registry):
        self.settings = settings
        self.store = store
        self.registry = registry
        self.doctor = DoctorCache(settings)
        self.scores = EpisodeScoreCache(settings, store)
        self.recycle = RecycleBin(settings, store, registry)

    def register(self, app, *, embedded):
        register_transport_routes(app, self, embedded=embedded)

    def service_ready(self):
        state = _gateway_state(self.store)
        return state is not None and state.get("state") == "READY"

    async def recycle_page(self, dataset_id):
        await asyncio.to_thread(self.recycle.resolve, f"bp/{dataset_id}")
        page = Path(__file__).with_name("viewer_ui") / "recycle.html"
        return HTMLResponse(
            themed_html(page.read_text()),
            headers={"Cache-Control": "no-store"},
        )

    async def bp_transport(self, request, dataset_id, path):
        preview = self.registry.lookup(f"bp/{dataset_id}")
        if preview is None:
            return JSONResponse(
                {"detail": "预览已过期或不存在，请在管理台重新打开"},
                status_code=404,
                headers={"Cache-Control": "no-store"},
            )
        episode = path.removeprefix("episode_")
        if (
            path.startswith("episode_")
            and episode.isdigit()
            and self.service_ready()
        ):
            status = await asyncio.to_thread(
                self.recycle.status, f"bp/{dataset_id}"
            )
            if int(episode) >= status["total_episodes"] or any(
                entry["episode"] == int(episode)
                for entry in status["entries"]
            ):
                return RedirectResponse(
                    f"/bp/{dataset_id}/trash",
                    status_code=303,
                    headers={"Cache-Control": "no-store"},
                )
        if path.startswith("resolve/main/"):
            return serve_dataset_file(
                request,
                self.registry,
                f"bp/{dataset_id}/{path}",
                preview=preview,
            )
        return await self.proxy_ui(request, f"bp/{dataset_id}/{path}")

    async def next_asset(self, request, path):
        return await self.proxy_ui(request, f"_next/{path}")

    async def data(self, request, path):
        rest = path[len("_data/") :] if path.startswith("_data/") else path
        route = rest.strip("/")
        if route in {"doctor", "doctor/report.json"}:
            repo_id = request.query_params.get("dataset", "")
            row = self.registry.lookup(repo_id)
            if row is None:
                return JSONResponse(
                    {"detail": "预览已过期或不存在，请在管理台重新打开"},
                    status_code=404,
                )
            payload = await asyncio.to_thread(self.doctor.payload, row)
            if route.endswith("report.json"):
                return JSONResponse(
                    payload, headers={"Cache-Control": "no-store"}
                )
            return HTMLResponse(
                _doctor_html(payload, row["dataset_path"], "up to 20 episodes"),
                headers={"Cache-Control": "no-store"},
            )
        return serve_dataset_file(request, self.registry, rest)

    async def proxy_ui(self, request, path):
        if not self.service_ready():
            if request.headers.get("accept", "").startswith("text/html"):
                return HTMLResponse(
                    STARTING_PAGE, headers={"Cache-Control": "no-store"}
                )
            return JSONResponse(
                {"detail": "预览仍在启动，请稍后刷新"}, status_code=503
            )
        url = f"http://127.0.0.1:{self.settings.viewer_ui_port}/{path}"
        if request.url.query:
            url += "?" + request.url.query
        return await proxy_next_response(request, url)

    def close(self):
        self.scores.close()


def register_transport_routes(app: FastAPI, transport, *, embedded: bool) -> None:
    @app.get("/bp/{dataset_id}/trash")
    async def recycle_page(dataset_id: str):
        return await transport.recycle_page(dataset_id)

    @app.api_route("/bp/{dataset_id}/{path:path}", methods=["GET", "HEAD"])
    async def bp_transport(request: Request, dataset_id: str, path: str):
        return await transport.bp_transport(request, dataset_id, path)

    @app.api_route("/_next/{path:path}", methods=["GET", "HEAD"])
    async def next_asset(request: Request, path: str):
        return await transport.next_asset(request, path)

    @app.api_route("/_data/{path:path}", methods=["GET", "HEAD"])
    async def data(request: Request, path: str):
        return await transport.data(request, path)

    @app.api_route("/favicon.ico", methods=["GET", "HEAD"])
    async def favicon(request: Request):
        return await transport.proxy_ui(request, "favicon.ico")

    @app.get("/bodypipeline-icon.png")
    @app.get("/bodypipeline-mark.png")
    def bodypipeline_asset(request: Request):
        return FileResponse(ASSETS / request.url.path.removeprefix("/"))

    if embedded:
        return

    @app.get("/")
    async def index():
        rows = transport.registry.active()
        if not rows:
            return JSONResponse(
                {"detail": "预览尚未就绪或已过期，请在管理台重新打开"},
                status_code=503,
            )
        return RedirectResponse(
            rows[0]["page_path"],
            status_code=307,
            headers={"Cache-Control": "no-store"},
        )


__all__ = [
    "VIEWER_ASSET_PATHS",
    "VIEWER_PROXY_PREFIXES",
    "DoctorCache",
    "ManualTrimRequest",
    "RecycleSelection",
    "SERVICE_KEY",
    "SERVICE_KIND",
    "STARTING_PAGE",
    "ViewerTransport",
    "embedded_enabled",
    "register_transport_routes",
]
