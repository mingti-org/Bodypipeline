# Embedded Viewer Gateway Safe Cutover Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve the existing Visualizer through port 8081 with an explicit route allowlist while keeping the workbench and standalone 8082 behavior intact.

**Architecture:** Extract the Visualizer page/data transport into a shared module used by both `create_gateway()` and `create_app()`. Add a default-off embedding flag, reuse the Server's persisted preview state, and move the few Gateway-only APIs into native Server handlers so no Visualizer catch-all can shadow `/api/*`, `/assets/*`, uploads, or workbench pages.

**Tech Stack:** Python 3.12, FastAPI, Starlette/TestClient, httpx, pytest, Next.js subprocess, HTTP Range streaming

**Spec:** `docs/superpowers/specs/2026-11-20-embedded-viewer-user-execution-design.md`

## Global Constraints

- Authenticated identity comes only from `request.state.principal`.
- Usernames, UIDs, SSH execution handles, and login nodes are never accepted from request JSON, query parameters, or client-controlled headers.
- A personal execution job stores `owner_user_id`, `owner_uid`, `submitted_as`, `slurm_user`, `slurm_execution_id`, and `submission_auth` from `Principal.job_metadata()`.
- `slurm_execution_id` is an internal execution handle and never appears in API responses, browser state, UI code, logs, error messages, receipts, or checkout artifacts.
- A missing or expired personal SSH session fails closed and never falls back to the Server account or another user.
- Service-token, device-token, and internal jobs retain existing `service-account` behavior; legacy jobs without `slurm_execution_id` remain resumable.
- The embedded application must not contain a Visualizer `/{path:path}` catch-all.
- `/api/*`, `/assets/*`, uploads, workbench pages, and unknown paths are never forwarded to Next.js.
- Next.js proxy requests accept only `GET` and `HEAD` and never forward `Authorization` or `Cookie`.
- The Server root remains the workbench entry point in embedded mode.
- `viewer_gateway.embedded` defaults to `false`; unknown keys and non-boolean values are configuration errors.
- The 8082 service remains available throughout the first cutover and is removed only by a later release.
- This plan changes gateway routing only. Task execution identity is implemented by `docs/superpowers/plans/2026-11-20-viewer-user-execution-identity.md` after this plan is accepted.

---

### Task 1: Add the strict embedding configuration

**Files:**
- Create: `pipeline/server/viewer_gateway.py`
- Modify: `pipeline/shared/config.py:88-170`
- Modify: `pipeline/shared/config.py:180-235`
- Test: `tests/test_viewer.py`

**Interfaces:**
- Produces: `Settings.viewer_gateway: dict`
- Produces: `embedded_enabled(settings) -> bool`
- Consumes: deployment JSON under the `viewer_gateway` key

- [ ] **Step 1: Write the failing configuration tests**

```python
import pytest


def test_viewer_gateway_configuration_is_strict(settings_factory):
    assert settings_factory("server").viewer_gateway == {}
    assert settings_factory(
        "server", viewer_gateway={"embedded": True}
    ).viewer_gateway == {"embedded": True}

    with pytest.raises(ValueError, match="viewer_gateway"):
        settings_factory("server", viewer_gateway={"embedded": 1})

    with pytest.raises(ValueError, match="viewer_gateway"):
        settings_factory("server", viewer_gateway={"unknown": True})
```

- [ ] **Step 2: Run the focused test and confirm it fails**

Run:

```bash
pytest tests/test_viewer.py::test_viewer_gateway_configuration_is_strict -v
```

Expected: FAIL because `Settings` does not define or validate
`viewer_gateway`.

- [ ] **Step 3: Add the setting and validation**

```python
# pipeline/shared/config.py
viewer_gateway: dict = field(default_factory=dict)


def _validate_viewer_gateway_configuration(self):
    if not isinstance(self.viewer_gateway, dict):
        raise ValueError("viewer_gateway 必须是对象")
    unknown = set(self.viewer_gateway) - {"embedded"}
    if unknown:
        raise ValueError(
            f"viewer_gateway 包含未知字段：{', '.join(sorted(unknown))}"
        )
    if (
        "embedded" in self.viewer_gateway
        and type(self.viewer_gateway["embedded"]) is not bool
    ):
        raise ValueError("viewer_gateway.embedded 必须为布尔值")
```

Call `_validate_viewer_gateway_configuration()` from
`Settings.__post_init__()`. Add the helper below `__post_init__`.

```python
# pipeline/server/viewer_gateway.py
def embedded_enabled(settings) -> bool:
    config = getattr(settings, "viewer_gateway", {})
    return isinstance(config, dict) and config.get("embedded") is True
```

- [ ] **Step 4: Run the focused test and verify it passes**

Run:

```bash
pytest tests/test_viewer.py::test_viewer_gateway_configuration_is_strict -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/shared/config.py pipeline/server/viewer_gateway.py tests/test_viewer.py
git commit -m "feat(server): add strict viewer gateway configuration"
```

### Task 2: Extract an explicit Visualizer transport

**Files:**
- Modify: `pipeline/server/viewer_gateway.py`
- Modify: `pipeline/server/viewer.py:443-830`
- Test: `tests/test_viewer.py`

**Interfaces:**
- Produces: `VIEWER_PROXY_PREFIXES: tuple[str, ...]`
- Produces: `ViewerTransport(settings, store, registry)`
- Produces: `ViewerTransport.register(app, *, embedded: bool) -> None`
- Produces: `ViewerTransport.close() -> None`
- Produces: `ViewerTransport.service_ready() -> bool`
- Produces: `ViewerTransport.recycle_page(dataset_id) -> Response`
- Produces: `ViewerTransport.data(request, path) -> Response`
- Produces: `ViewerTransport.proxy_ui(request, path) -> Response`
- Produces: `ViewerTransport.scores: EpisodeScoreCache`
- Produces: `ViewerTransport.recycle: RecycleBin`
- Produces: `RecycleSelection`, moved from `viewer.py`
- Consumes: `PreviewRegistry`, `ViewerManager`, `Store`, and the persisted
  `viewer/service` state

- [ ] **Step 1: Write the failing no-catch-all test**

```python
def test_standalone_gateway_does_not_proxy_unowned_paths(
    settings_factory, monkeypatch
):
    import httpx

    settings = settings_factory("viewer")
    store = Store(settings.root / "state")
    store.put("viewer", "service", {"state": "READY"})
    calls = []

    async def upstream(request):
        calls.append(request.url.path)
        return httpx.Response(200, text="must not be called")

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        "pipeline.server.viewer_gateway.httpx.AsyncClient",
        lambda **kwargs: real_client(
            transport=httpx.MockTransport(upstream), **kwargs
        ),
    )
    with TestClient(create_gateway(settings)) as client:
        client.post("/api/login", json={"token": settings.token})
        response = client.get("/not-a-viewer-route")

    assert response.status_code == 404
    assert calls == []
```

- [ ] **Step 2: Run the test and confirm the current catch-all fails it**

Run:

```bash
pytest tests/test_viewer.py::test_standalone_gateway_does_not_proxy_unowned_paths -v
```

Expected: FAIL because the current `/{path:path}` route sends the unknown path
to Next.js or returns the 503 startup page.

- [ ] **Step 3: Implement the shared transport route set**

Use this prefix contract:

```python
VIEWER_PROXY_PREFIXES = ("/bp/", "/_next/")
VIEWER_ASSET_PATHS = (
    "/favicon.ico",
    "/bodypipeline-icon.png",
    "/bodypipeline-mark.png",
)
```

Move these existing units from `pipeline/server/viewer.py` into
`pipeline/server/viewer_gateway.py` before creating `ViewerTransport`:

- `STARTING_PAGE`
- `_doctor_html()`
- `DoctorCache`
- `SERVICE_KEY`, `SERVICE_KIND`, and `_gateway_state()`
- `ManualTrimRequest`
- `RecycleSelection`

Change `viewer.py` to import the moved
`SERVICE_KEY`/`SERVICE_KIND` constants, request models, and transport objects
from `viewer_gateway.py`. Do not import `viewer.py` from `viewer_gateway.py`;
that would create a circular import during module startup.

Implement the service object with complete route responsibilities:

```python
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
        episode = path.removeprefix("episode_")
        if path.startswith("episode_") and episode.isdigit():
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
            )
        return await self.proxy_ui(request, f"bp/{dataset_id}/{path}")

    async def next_asset(self, request, path):
        return await self.proxy_ui(request, f"_next/{path}")

    async def data(self, request, path):
        rest = path[len("_data/"):] if path.startswith("_data/") else ""
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
```

`proxy_next_response()` is the existing streaming block moved verbatim from
`viewer.py`: it forwards only `FORWARDED_UI_HEADERS`, rejects connection errors
with `DomainError("预览后端不可达", 502)`, streams raw bytes, and returns only
the existing `content-type`, `content-length`, `content-range`,
`accept-ranges`, `content-encoding`, `etag`, `vary`, `location`, and
`cache-control` response headers.

`register_transport_routes()` must register these exact handlers:

```python
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
```

The `bp_transport()` handler redirects trashed episodes, sends
`/bp/{dataset_id}/resolve/main/...` requests to `serve_dataset_file()`, and
proxies only Visualizer page paths. The `next_asset()` handler always uses the
same proxy function. The proxy function forwards only the existing audited
headers:

```python
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
```

Keep the existing startup page, recycle redirect, range validation, streaming,
and response-header behavior. Register the standalone root redirect only when
`embedded=False`.

Import the moved `RecycleSelection` model from `viewer_gateway.py` in both
`viewer.py` and `app.py`; do not define two request models with the same name.
Update the existing Doctor test monkeypatch from
`pipeline.server.viewer.subprocess.run` to
`pipeline.server.viewer_gateway.subprocess.run` because `DoctorCache` now lives
in the shared transport module.

Update `create_gateway()` to construct one `ViewerTransport`, assign both
`app.state.viewer_transport = transport` and
`app.state.episode_scores = transport.scores`, then call
`transport.register(app, embedded=False)`. Keep the standalone API proxy routes
in `viewer.py`; they call the transport's `recycle`, `scores`, and registry
attributes. The standalone lifespan closes `transport`, so it must not also
close `transport.scores`.

- [ ] **Step 4: Run all standalone gateway tests**

Run:

```bash
pytest tests/test_viewer.py tests/test_instruction_api.py tests/test_episode_scores.py -q
```

Expected: PASS, including root redirect, `/_data` ranges, `/_next` proxy,
unknown-path 404, recycle, score, Doctor, and Visualizer API proxy tests.

- [ ] **Step 5: Commit**

```bash
git add pipeline/server/viewer.py pipeline/server/viewer_gateway.py tests/test_viewer.py
git commit -m "refactor(server): extract explicit viewer transport"
```

### Task 3: Register the Visualizer transport in the Server behind the flag

**Files:**
- Modify: `pipeline/server/app.py:217-410`
- Modify: `pipeline/server/viewer_gateway.py`
- Test: `tests/test_viewer.py`

**Interfaces:**
- Consumes: `embedded_enabled(settings)`
- Consumes: `ViewerManager.registry`
- Produces: `app.state.viewer_transport`
- Produces: embedded `/bp/*`, `/_next/*`, `/_data/*`, and Visualizer asset routes

- [ ] **Step 1: Add the failing embedded-routing test**

```python
def test_server_embeds_only_allowlisted_viewer_routes(
    settings_factory, tmp_path, monkeypatch
):
    import httpx

    settings = settings_factory(
        "server",
        viewer_gateway={"embedded": True},
        viewer_origin="http://testserver",
    )
    app = create_app(settings, background=False)
    app.state.viewer.registry.register("job-a", make_dataset(tmp_path, "viewer"))
    app.state.ingest.store.put("viewer", "service", {"state": "READY"})
    seen = []

    async def upstream(request):
        seen.append(str(request.url))
        return httpx.Response(200, text="<html>viewer</html>")

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        "pipeline.server.viewer_gateway.httpx.AsyncClient",
        lambda **kwargs: real_client(
            transport=httpx.MockTransport(upstream), **kwargs
        ),
    )
    headers = {"Authorization": "Bearer " + settings.token}

    with TestClient(app, headers=headers) as client:
        page = client.get("/bp/job-a/episode_0")
        assert page.status_code == 200
        assert page.text == "<html>viewer</html>"
        assert (
            client.get(
                "/_data/bp/job-a/resolve/main/videos/a.mp4"
            ).status_code
            == 200
        )
        assert client.get("/assets/app.js").status_code == 200
        assert client.get("/api/config").json()["role"] == "server"
        root = client.get("/")
        assert root.status_code == 200
        assert "text/html" in root.headers["content-type"]
        assert client.get("/not-owned-by-viewer").status_code == 404

    assert seen == [
        f"http://127.0.0.1:{settings.viewer_ui_port}/bp/job-a/episode_0"
    ]
```

Add the opt-in regression:

```python
def test_server_viewer_transport_is_opt_in(settings_factory):
    settings = settings_factory("server")
    app = create_app(settings, background=False)
    with TestClient(
        app, headers={"Authorization": "Bearer " + settings.token}
    ) as client:
        assert client.get("/bp/job-a/episode_0").status_code == 404
        assert client.get("/api/config").status_code == 200
```

- [ ] **Step 2: Run the tests and confirm the embedded route is missing**

Run:

```bash
pytest tests/test_viewer.py::test_server_embeds_only_allowlisted_viewer_routes tests/test_viewer.py::test_server_viewer_transport_is_opt_in -v
```

Expected: the opt-in test passes; the embedded test fails with 404 for the
Visualizer page because `create_app()` does not register the transport.

- [ ] **Step 3: Register the transport and wire its lifecycle**

Construct the transport only after the `FastAPI` instance exists. Keep the
existing lifespan closure above the construction; it resolves
`viewer_transport` at shutdown after the variable has been assigned.

```python
# Existing line:
app = FastAPI(title="Embodied Pipeline Server", lifespan=lifespan)

viewer_transport = ViewerTransport(
    settings,
    ingest.store,
    viewer.registry,
)
app.state.ingest = ingest
app.state.runner = runner
app.state.viewer = viewer
app.state.viewer_transport = viewer_transport
app.state.episode_scores = viewer_transport.scores
```

Keep the other existing `app.state.*` assignments. Call `install_security()`
first, then register the embedded transport:

```python
install_security(
    app,
    settings,
    device_prefix="/device/",
    user_sessions=user_sessions,
)
if embedded_enabled(settings):
    viewer_transport.register(app, embedded=True)
```

Register so `/_data`, `/bp`, and `/_next` use the Server cookie and bearer
authentication. Register before `mount_ui()` only for these non-overlapping
route families. Do not register its root redirect in embedded mode.

Add `viewer_transport.close()` to the lifespan shutdown chain after
`viewer.stop()` and before `user_sessions.close_all()`. Do not close it twice
from `create_gateway()`.

- [ ] **Step 4: Run embedded and standalone gateway tests**

Run:

```bash
pytest tests/test_viewer.py tests/test_episode_scores.py tests/test_instruction_api.py -q
```

Expected: PASS in both modes. The Server root remains the workbench, unknown
paths return 404, and standalone 8082 still redirects `/` to its newest preview.

- [ ] **Step 5: Commit**

```bash
git add pipeline/server/app.py pipeline/server/viewer_gateway.py tests/test_viewer.py
git commit -m "feat(server): embed allowlisted viewer transport"
```

### Task 4: Add native Server handlers for Gateway-only APIs

**Files:**
- Modify: `pipeline/server/app.py:760-830`
- Modify: `pipeline/server/app.py:1140-1210`
- Modify: `pipeline/server/viewer_gateway.py`
- Test: `tests/test_stationary_trim.py`
- Test: `tests/test_episode_scores.py`
- Test: `tests/test_recycle.py`

**Interfaces:**
- Produces: `GET|POST /api/episode-scores/{dataset_id}`
- Produces: `POST /api/manual-trims/{dataset_id}`
- Produces: `GET /api/manual-trims/{dataset_id}/jobs/{job_id}`
- Produces: `POST /api/manual-trims/{dataset_id}/jobs/{job_id}/preview`
- Produces: `GET /api/recycle/{dataset_id}`
- Produces: `POST /api/recycle/{dataset_id}/{action}`
- Consumes: `app.state.viewer_transport.registry`, `.scores`, and `.recycle`

- [ ] **Step 1: Write the failing native endpoint tests**

```python
def test_embedded_server_owns_gateway_only_apis(
    settings_factory, monkeypatch
):
    settings = settings_factory(
        "server",
        letools_bin=sys.executable,
        viewer_gateway={"embedded": True},
        viewer_origin="http://testserver",
    )
    app = create_app(settings, background=False)
    seed(app)
    app.state.viewer.registry.register("source", settings.root / "source")
    headers = {"Authorization": "Bearer " + settings.token}

    monkeypatch.setattr(
        app.state.viewer_transport.scores,
        "start",
        lambda preview, *, principal=None: {"state": "QUEUED"},
    )
    with TestClient(app, headers=headers) as client:
        score = client.post("/api/episode-scores/source")
        assert score.status_code == 202
        assert score.json() == {"state": "QUEUED"}
        recycle = client.get("/api/recycle/source")
        assert recycle.status_code == 200
        trim = client.post(
            "/api/manual-trims/source",
            json={"episode_index": 0, "remove_ranges": [[1, 2]]},
        )
        assert trim.status_code == 202, trim.text
        assert trim.json()["operation"] == "manual_trim"
```

Add unknown-preview assertions:

```python
@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("POST", "/api/episode-scores/missing", None),
        ("GET", "/api/recycle/missing", None),
        (
            "POST",
            "/api/manual-trims/missing",
            {"episode_index": 0, "remove_ranges": [[1, 2]]},
        ),
    ],
)
def test_embedded_apis_require_registered_preview(
    settings_factory, method, path, body
):
    settings = settings_factory(
        "server",
        letools_bin=sys.executable,
        viewer_gateway={"embedded": True},
    )
    app = create_app(settings, background=False)
    headers = {"Authorization": "Bearer " + settings.token}
    with TestClient(app, headers=headers) as client:
        response = client.request(method, path, json=body)
    assert response.status_code == 404
```

- [ ] **Step 2: Run the tests and confirm the routes are absent**

Run:

```bash
pytest tests/test_stationary_trim.py::test_embedded_server_owns_gateway_only_apis tests/test_stationary_trim.py::test_embedded_apis_require_registered_preview -v
```

Expected: FAIL with 404 because those API routes exist only in
`create_gateway()`.

- [ ] **Step 3: Implement the native routes with shared helpers**

Import `ManualTrimRequest`, `RecycleSelection`, and `ViewerTransport` from
`pipeline.server.viewer_gateway`. Change the FastAPI response import to:

```python
from fastapi.responses import FileResponse, JSONResponse
```

Remove the duplicate `ManualTrimRequest` and `RecycleSelection` definitions
from both `app.py` and `viewer.py`.

Add the shared preview resolver:

```python
def registered_preview(transport: ViewerTransport, dataset_id: str) -> dict:
    row = transport.registry.lookup(f"bp/{dataset_id}")
    if row is None:
        raise DomainError(
            "预览已过期或不存在，请在管理台重新打开",
            404,
        )
    return row
```

Place the following route definitions inside one
`if embedded_enabled(settings):` block. This is required so the routes remain
404 when the feature flag is off:

```python
@app.get("/api/episode-scores/{dataset_id}")
def embedded_score_status(dataset_id: str):
    row = registered_preview(viewer_transport, dataset_id)
    result = viewer_transport.scores.status(row)
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


@app.post("/api/episode-scores/{dataset_id}", status_code=202)
def embedded_score_start(dataset_id: str):
    row = registered_preview(viewer_transport, dataset_id)
    result = viewer_transport.scores.start(row)
    return JSONResponse(result, status_code=202)


@app.get("/api/recycle/{dataset_id}")
def embedded_recycle_status(dataset_id: str):
    registered_preview(viewer_transport, dataset_id)
    return viewer_transport.recycle.status(f"bp/{dataset_id}")


@app.post("/api/recycle/{dataset_id}/{action}")
def embedded_recycle_change(
    dataset_id: str, action: str, body: RecycleSelection
):
    registered_preview(viewer_transport, dataset_id)
    if action not in {"trash", "restore", "purge"}:
        raise DomainError("不支持的回收站操作", 404)
    return viewer_transport.recycle.change(
        f"bp/{dataset_id}",
        action,
        sorted(set(body.episodes)),
        body.revision,
    )
```

Implement the manual trim handlers directly against the Server's persisted
state. Do not call the Server's own loopback HTTP API.

```python
@app.post("/api/manual-trims/{dataset_id}", status_code=202)
def embedded_manual_trim_start(
    dataset_id: str, body: ManualTrimRequest
):
    row = registered_preview(viewer_transport, dataset_id)
    job = trims.enqueue_manual(
        row["job_id"],
        body.episode_index,
        body.remove_ranges,
        name=body.name,
        run_doctor=body.run_doctor,
        in_place=body.in_place,
    )
    return job_response(job)


def manual_trim_job(dataset_id: str, job_id: str) -> dict:
    preview = registered_preview(viewer_transport, dataset_id)
    if not job_id or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in job_id
    ):
        raise DomainError("裁剪任务编号无效", 404)
    try:
        job = ingest.store.get("job", job_id)
    except DomainError as exc:
        if exc.code == 404:
            raise DomainError("裁剪任务不存在", 404) from exc
        raise
    if (
        job.get("operation") != "manual_trim"
        or job.get("source_job_id") != preview["job_id"]
    ):
        raise DomainError("裁剪任务不存在", 404)
    return job


@app.get("/api/manual-trims/{dataset_id}/jobs/{job_id}")
def embedded_manual_trim_status(dataset_id: str, job_id: str):
    return job_response(manual_trim_job(dataset_id, job_id))


@app.post("/api/manual-trims/{dataset_id}/jobs/{job_id}/preview")
def embedded_manual_trim_preview(dataset_id: str, job_id: str):
    job = manual_trim_job(dataset_id, job_id)
    if job.get("state") not in {
        "COMPLETED",
        "CONVERTED",
        "SYNTHETIC_READY",
        "QUARANTINED",
    }:
        raise DomainError("裁剪尚未完成，暂时无法预览", 409)
    with runner.lock:
        opened = viewer.open(ingest.store.get("job", job_id))
    return {"url": opened.get("url")}
```

Keep plan-one behavior as `principal=None`; the identity plan changes the
`enqueue_manual()` call after this gateway plan is accepted.

Do not register duplicate `/api/instructions`, `/api/task-templates`, or
`/api/subtask-review` handlers. The existing Server routes remain authoritative.

- [ ] **Step 4: Run the focused API and regression tests**

Run:

```bash
pytest tests/test_stationary_trim.py tests/test_episode_scores.py tests/test_recycle.py tests/test_instruction_api.py tests/test_subtask_review.py -q
```

Expected: PASS. Embedded requests reach native Server handlers and do not make
recursive loopback HTTP calls to 8081.

- [ ] **Step 5: Commit**

```bash
git add pipeline/server/app.py pipeline/server/viewer_gateway.py tests/test_stationary_trim.py tests/test_episode_scores.py tests/test_recycle.py
git commit -m "feat(server): add native embedded viewer APIs"
```

### Task 5: Prove route, authentication, and streaming isolation

**Files:**
- Modify: `pipeline/server/viewer_gateway.py`
- Modify: `tests/test_viewer.py`
- Modify: `tests/test_user_identity.py`

**Interfaces:**
- Consumes: Server `pipeline_server` HttpOnly cookie
- Produces: no Visualizer-specific browser login in embedded mode
- Produces: exact proxy methods and forwarding headers

- [ ] **Step 1: Add the failing same-origin identity test**

```python
def test_embedded_viewer_uses_server_cookie_and_never_forwards_it(
    settings_factory, tmp_path, monkeypatch
):
    import httpx

    settings = settings_factory(
        "server",
        viewer_gateway={"embedded": True},
        viewer_origin="http://testserver",
        user_sessions={
            "enabled": True,
            "login_nodes": ["H800-node11"],
        },
    )
    app = create_app(settings, background=False)
    app.state.viewer.registry.register("job-a", make_dataset(tmp_path, "viewer"))
    app.state.ingest.store.put("viewer", "service", {"state": "READY"})
    monkeypatch.setattr(
        app.state.user_sessions,
        "principal",
        lambda cookie: Principal(username="alice", uid=1001)
        if cookie == "opaque-user-cookie"
        else None,
    )
    seen = []

    async def upstream(request):
        seen.append(request)
        return httpx.Response(200, text="ok")

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        "pipeline.server.viewer_gateway.httpx.AsyncClient",
        lambda **kwargs: real_client(
            transport=httpx.MockTransport(upstream), **kwargs
        ),
    )
    with TestClient(app) as client:
        client.cookies.set(
            "pipeline_server", "opaque-user-cookie", domain="testserver"
        )
        response = client.get("/bp/job-a/episode_0")

    assert response.status_code == 200
    assert "cookie" not in seen[0].headers
    assert "authorization" not in seen[0].headers
```

Add mutation-origin coverage:

```python
def test_embedded_viewer_write_requires_same_origin(
    settings_factory, tmp_path, monkeypatch
):
    settings = settings_factory(
        "server",
        viewer_gateway={"embedded": True},
        user_sessions={"enabled": True, "login_nodes": ["H800-node11"]},
    )
    app = create_app(settings, background=False)
    app.state.viewer.registry.register("job-a", make_dataset(tmp_path, "viewer"))
    monkeypatch.setattr(
        app.state.user_sessions,
        "principal",
        lambda cookie: Principal(username="alice", uid=1001)
        if cookie == "opaque-user-cookie"
        else None,
    )
    with TestClient(app) as client:
        client.cookies.set(
            "pipeline_server", "opaque-user-cookie", domain="testserver"
        )
        response = client.post(
            "/api/episode-scores/job-a",
            headers={"Origin": "https://untrusted.example"},
        )
    assert response.status_code == 403
```

- [ ] **Step 2: Run the tests and inspect failures**

Run:

```bash
pytest tests/test_viewer.py::test_embedded_viewer_uses_server_cookie_and_never_forwards_it tests/test_viewer.py::test_embedded_viewer_write_requires_same_origin -v
```

Expected: the read test fails if the transport still has a catch-all or
forwards credentials. The write test fails if native API routes bypass the
Server security middleware. Retain either test as regression coverage when it
already passes.

- [ ] **Step 3: Fix only the demonstrated isolation gap**

Use one proxy construction with the exact forwarding set:

```python
outgoing = {
    key: value
    for key, value in request.headers.items()
    if key.lower() in FORWARDED_UI_HEADERS
}
response = await client.send(
    client.build_request(request.method, url, headers=outgoing),
    stream=True,
    follow_redirects=False,
)
```

If a cookie-authenticated write lacks Origin validation, register the route on
the Server application so the existing `install_security()` middleware sees it.
Keep the existing 403 response.

Add range, HEAD, unknown preview, traversal, symlink, cross-dataset, and startup
page tests using the existing `tests/test_viewer.py` fixtures. The asserted
matrix is:

| Case | Expected |
| --- | --- |
| Registered `/_data` file | 200 |
| Single byte range | 206 with `Content-Range` |
| HEAD range | 206 with no body |
| Multi range | 416 |
| Unknown preview | 404 |
| `..`, encoded `..`, or symlink escape | 404 |
| `POST` to a data path | 405 after same-origin validation |
| `/api/*` not owned by viewer | Server response |
| `/assets/*` | Server workbench asset |
| Unknown root path | 404 |

- [ ] **Step 4: Run the route and security suites**

```bash
pytest tests/test_viewer.py tests/test_user_identity.py tests/test_login_password.py -q
npm test -- --run tests/viewer_instruction.test.ts
```

Expected: PASS. The embedded UI uses the Server cookie, all writes retain
same-origin checks, and the proxy forwards no credentials.

- [ ] **Step 5: Commit**

```bash
git add pipeline/server/viewer_gateway.py tests/test_viewer.py tests/test_user_identity.py
git commit -m "test(server): verify embedded viewer security boundary"
```

### Task 6: Document, validate, and gate the cutover

**Files:**
- Create: `docs/operations/viewer-gateway-cutover.md`
- Modify: `configs/server.deployed.example.json`
- Modify: `configs/server.json`
- Test: `tests/test_server_api.py`

**Interfaces:**
- Consumes: the exact JSON keys `viewer_gateway.embedded` and `viewer_origin`
- Produces: a repeatable enable, smoke, rollback, and observation procedure

- [ ] **Step 1: Add a failing deployment-config contract test**

```python
def test_deployed_server_config_defaults_viewer_gateway_off():
    config = json.loads(
        Path("configs/server.deployed.example.json").read_text(encoding="utf-8")
    )
    assert config["viewer_gateway"] == {"embedded": False}
```

- [ ] **Step 2: Run the test and confirm the key is absent**

```bash
pytest tests/test_server_api.py::test_deployed_server_config_defaults_viewer_gateway_off -v
```

Expected: FAIL with `KeyError: 'viewer_gateway'`.

- [ ] **Step 3: Add disabled defaults and the cutover runbook**

Add this object to `configs/server.deployed.example.json` and
`configs/server.json`:

```json
"viewer_gateway": {
  "embedded": false
}
```

The runbook must use this deployment sequence:

```text
1. Deploy the new release with embedded=false.
2. Confirm 8081 workbench and 8082 preview are unchanged.
3. Stop starting new Visualizer jobs during the cutover window.
4. Set embedded=true.
5. Set viewer_origin to the public 8081 origin.
6. Restart the Server release process once.
7. Run all smoke checks while 8082 remains running.
8. Roll out to users only after a non-admin H800 browser test passes.
9. Observe for one full work cycle before disabling 8082.
```

The runbook includes these health and route checks:

```bash
curl -fsS http://127.0.0.1:8081/healthz
curl -fsS -o /dev/null http://127.0.0.1:8081/
curl -fsS -o /dev/null http://127.0.0.1:8081/assets/app.js
curl -fsS -H "Authorization: Bearer $PIPELINE_TOKEN" \
  http://127.0.0.1:8081/api/config
```

In an authenticated browser, record the following evidence:

```text
1. `/` renders the workbench.
2. `/assets/app.js` returns the workbench client bundle.
3. `/api/listings/status?robot_type=<existing>` returns the pre-cutover result.
4. Opening one existing preview produces a URL on 8081.
5. `/bp/<job>/episode_0` renders the Visualizer.
6. `/_next/static/...` resources return 200.
7. A video `/_data/...` request returns 206 for a byte range.
8. Doctor, episode score, recycle, manual trim, instruction template,
   instruction episode edit, and review pages load.
9. An unknown path returns 404 and no workbench response is replaced by the
   Visualizer startup page.
10. The browser has only the `pipeline_server` cookie; no second login is
    requested.
```

Record the release identifier before and after cutover. The rollback section
must use this exact order:

```text
1. Set viewer_gateway.embedded=false.
2. Restore viewer_origin to the last known 8082 URL.
3. Restart the Server process.
4. Verify `/` still renders the workbench.
5. Open a new preview and confirm its URL uses 8082.
6. Confirm no process is listening on an unexpected new public port.
```

Include this identity caveat during the observation period: standalone 8082
remains a compatibility route for preview and token-authenticated automation;
the authenticated per-user execution guarantee applies to the embedded 8081
surface after the second plan is deployed.

- [ ] **Step 4: Run the full gateway, Server, and template regression suites**

```bash
pytest \
  tests/test_viewer.py \
  tests/test_server_api.py \
  tests/test_server_lists.py \
  tests/test_workbench_lists.py \
  tests/test_collections.py \
  tests/test_ingest.py \
  tests/test_stationary_trim.py \
  tests/test_episode_scores.py \
  tests/test_recycle.py \
  tests/test_instruction_api.py \
  tests/test_instruction_review.py \
  tests/test_subtask_review.py -q
ruff check pipeline/server/app.py pipeline/server/viewer.py pipeline/server/viewer_gateway.py pipeline/shared/config.py tests/test_viewer.py
```

Expected: PASS. If the optional LeTools fixture is unavailable, record the
skipped tests and rerun them in the deployment environment before enabling the
flag.

- [ ] **Step 5: Commit** 

```bash
git add \
  configs/server.json \
  configs/server.deployed.example.json \
  docs/operations/viewer-gateway-cutover.md \
  tests/test_server_api.py
git commit -m "docs(deploy): add viewer gateway cutover runbook"
```
