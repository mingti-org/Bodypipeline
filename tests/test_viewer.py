import json
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pipeline.server.app import create_app
from pipeline.server.viewer import ViewerManager, create_gateway
from pipeline.server.viewer_bootstrap import UPSTREAM, patch_cache
from pipeline.server.viewer_data import PreviewRegistry, repo_id_for
from pipeline.server.viewer_gateway import ViewerTransport
from pipeline.shared.core import DomainError, Store
from pipeline.shared.identity import Principal


def make_dataset(root, name="ds"):
    data = root / name
    (data / "meta").mkdir(parents=True)
    (data / "meta/info.json").write_text(json.dumps({"fps": 30}))
    (data / "videos").mkdir()
    (data / "videos/a.mp4").write_bytes(b"0123456789")
    return data


def make_recycle_dataset(root, name="recycle"):
    data = make_dataset(root, name)
    (data / "meta/info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "total_episodes": 3,
                "total_frames": 30,
                "fps": 30,
            }
        )
    )
    return data


def test_viewer_gateway_configuration_is_strict(settings_factory):
    assert settings_factory("server").viewer_gateway == {}
    assert settings_factory(
        "server", viewer_gateway={"embedded": True}
    ).viewer_gateway == {"embedded": True}

    with pytest.raises(ValueError, match="viewer_gateway"):
        settings_factory("server", viewer_gateway={"embedded": 1})

    with pytest.raises(ValueError, match="viewer_gateway"):
        settings_factory("server", viewer_gateway={"unknown": True})


def test_embedded_enabled_is_default_off(settings_factory):
    from pipeline.server.viewer_gateway import embedded_enabled

    assert embedded_enabled(settings_factory("server")) is False
    assert (
        embedded_enabled(settings_factory("server", viewer_gateway={"embedded": True}))
        is True
    )


def test_server_embeds_only_allowlisted_viewer_routes(
    settings_factory, tmp_path, monkeypatch
):
    import httpx

    settings = settings_factory(
        "server",
        import_roots=[tmp_path],
        viewer_gateway={"embedded": True},
        viewer_origin="http://testserver",
    )
    app = create_app(settings, background=False)
    app.state.viewer.registry.register(
        "job-a", make_recycle_dataset(tmp_path, "viewer")
    )
    seen = []

    async def upstream(request):
        seen.append(str(request.url))
        return httpx.Response(200, stream=httpx.ByteStream(b"<html>viewer</html>"))

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        "pipeline.server.viewer_gateway.httpx.AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(upstream), **kwargs),
    )
    headers = {"Authorization": "Bearer " + settings.token}

    with TestClient(app, headers=headers) as client:
        # Startup recovery intentionally interrupts any persisted READY viewer.
        app.state.ingest.store.put("viewer", "service", {"state": "READY"})
        page = client.get("/bp/job-a/episode_0")
        assert page.status_code == 200
        assert page.text == "<html>viewer</html>"
        # Next.js bootstraps the page with inline scripts. The Server CSP is
        # intentionally limited to Server-owned pages so the embedded Viewer
        # keeps the policy supplied by its own build.
        assert "Content-Security-Policy" not in page.headers
        assert (
            client.get("/_data/bp/job-a/resolve/main/videos/a.mp4").status_code == 200
        )
        assert client.get("/assets/app.js").status_code == 200
        assert client.get("/api/config").json()["role"] == "server"
        root = client.get("/")
        assert root.status_code == 200
        assert "text/html" in root.headers["content-type"]
        assert "script-src 'self'" in root.headers["Content-Security-Policy"]
        assert client.get("/not-owned-by-viewer").status_code == 404

    assert seen == [f"http://127.0.0.1:{settings.viewer_ui_port}/bp/job-a/episode_0"]


@pytest.mark.parametrize(
    ("service_state", "headers"),
    [
        ("READY", {}),
        ("STARTING", {"Accept": "text/html"}),
    ],
)
def test_embedded_missing_preview_never_reaches_upstream_or_startup(
    settings_factory, monkeypatch, service_state, headers
):
    import httpx

    settings = settings_factory(
        "server",
        viewer_gateway={"embedded": True},
        viewer_origin="http://testserver",
    )
    app = create_app(settings, background=False)
    upstream_calls = []

    async def upstream(request):
        upstream_calls.append(request)
        return httpx.Response(200, stream=httpx.ByteStream(b"unexpected"))

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        "pipeline.server.viewer_gateway.httpx.AsyncClient",
        lambda **kwargs: real_client(
            transport=httpx.MockTransport(upstream), **kwargs
        ),
    )
    auth = {"Authorization": "Bearer " + settings.token}
    with TestClient(app) as client:
        app.state.ingest.store.put(
            "viewer", "service", {"state": service_state}
        )
        response = client.get(
            "/bp/missing/episode_0",
            headers={**auth, **headers},
        )

    assert response.status_code == 404
    assert "正在启动 LeRobot Visualizer" not in response.text
    assert upstream_calls == []


def test_embedded_viewer_uses_server_cookie_and_never_forwards_it(
    settings_factory, tmp_path, monkeypatch
):
    import httpx

    settings = settings_factory(
        "server",
        import_roots=[tmp_path],
        viewer_gateway={"embedded": True},
        viewer_origin="http://testserver",
        user_sessions={
            "enabled": True,
            "login_nodes": ["H800-node11"],
        },
    )
    app = create_app(settings, background=False)
    app.state.viewer.registry.register(
        "job-a", make_recycle_dataset(tmp_path, "viewer")
    )
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
        return httpx.Response(200, stream=httpx.ByteStream(b"ok"))

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        "pipeline.server.viewer_gateway.httpx.AsyncClient",
        lambda **kwargs: real_client(
            transport=httpx.MockTransport(upstream), **kwargs
        ),
    )
    with TestClient(app) as client:
        app.state.ingest.store.put("viewer", "service", {"state": "READY"})
        client.cookies.set("pipeline_server", "opaque-user-cookie")
        response = client.get("/bp/job-a/episode_0")

    assert response.status_code == 200
    assert (
        str(seen[0].url)
        == f"http://127.0.0.1:{settings.viewer_ui_port}/bp/job-a/episode_0"
    )
    assert "cookie" not in seen[0].headers
    assert "authorization" not in seen[0].headers


def test_embedded_viewer_requires_server_resolved_identity(
    settings_factory, tmp_path, monkeypatch
):
    import httpx

    settings = settings_factory(
        "server",
        import_roots=[tmp_path],
        viewer_gateway={"embedded": True},
        user_sessions={"enabled": True, "login_nodes": ["H800-node11"]},
    )
    app = create_app(settings, background=False)
    app.state.viewer.registry.register(
        "job-a", make_recycle_dataset(tmp_path, "viewer")
    )
    monkeypatch.setattr(app.state.user_sessions, "principal", lambda cookie: None)
    upstream_calls = []

    async def upstream(request):
        upstream_calls.append(request)
        return httpx.Response(200, stream=httpx.ByteStream(b"unexpected"))

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        "pipeline.server.viewer_gateway.httpx.AsyncClient",
        lambda **kwargs: real_client(
            transport=httpx.MockTransport(upstream), **kwargs
        ),
    )
    with TestClient(app) as client:
        app.state.ingest.store.put("viewer", "service", {"state": "READY"})
        client.cookies.set("pipeline_server", "opaque-user-cookie")
        response = client.get("/bp/job-a/episode_0")

    assert response.status_code == 401
    assert response.json() == {"detail": "请先登录或提供有效凭据"}
    assert "opaque-user-cookie" not in response.text
    assert "alice" not in response.text
    assert upstream_calls == []


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
        client.cookies.set("pipeline_server", "opaque-user-cookie")
        response = client.post(
            "/api/episode-scores/job-a",
            headers={"Origin": "https://untrusted.example"},
        )
    assert response.status_code == 403


def test_embedded_viewer_transport_rejects_cookie_authenticated_posts(
    settings_factory, tmp_path, monkeypatch
):
    import httpx

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
    upstream_calls = []

    async def upstream(request):
        upstream_calls.append(request)
        return httpx.Response(200, stream=httpx.ByteStream(b"unexpected"))

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        "pipeline.server.viewer_gateway.httpx.AsyncClient",
        lambda **kwargs: real_client(
            transport=httpx.MockTransport(upstream), **kwargs
        ),
    )
    origin = {"Origin": "http://testserver"}
    with TestClient(app) as client:
        client.cookies.set("pipeline_server", "opaque-user-cookie")
        app.state.ingest.store.put("viewer", "service", {"state": "READY"})
        page_write = client.post(
            "/bp/job-a/episode_0",
            headers=origin,
        )
        next_write = client.post(
            "/_next/static/chunk.js",
            headers=origin,
        )

    assert page_write.status_code == 405
    assert next_write.status_code == 405
    assert upstream_calls == []


def test_embedded_viewer_does_not_proxy_unknown_api_path(
    settings_factory, tmp_path, monkeypatch
):
    import httpx

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
    upstream_calls = []

    async def upstream(request):
        upstream_calls.append(request)
        return httpx.Response(200, stream=httpx.ByteStream(b"unexpected"))

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        "pipeline.server.viewer_gateway.httpx.AsyncClient",
        lambda **kwargs: real_client(
            transport=httpx.MockTransport(upstream), **kwargs
        ),
    )
    with TestClient(app) as client:
        client.cookies.set("pipeline_server", "opaque-user-cookie")
        response = client.get("/api/not-owned")

    assert response.status_code == 404
    assert upstream_calls == []


def test_embedded_viewer_proxy_forwards_exact_range_and_head_requests(
    settings_factory, tmp_path, monkeypatch
):
    import httpx

    settings = settings_factory(
        "server",
        import_roots=[tmp_path],
        viewer_gateway={"embedded": True},
        user_sessions={"enabled": True, "login_nodes": ["H800-node11"]},
    )
    app = create_app(settings, background=False)
    app.state.viewer.registry.register(
        "job-a", make_recycle_dataset(tmp_path, "viewer")
    )
    seen = []

    async def upstream(request):
        seen.append(request)
        content = b"2345" if request.method == "GET" else b"ignored"
        return httpx.Response(
            206,
            stream=httpx.ByteStream(content),
            headers={
                "Content-Range": "bytes 2-5/10"
                if request.method == "GET"
                else "bytes 0-3/4",
                "Content-Length": "4",
                "Accept-Ranges": "bytes",
            },
        )

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        "pipeline.server.viewer_gateway.httpx.AsyncClient",
        lambda **kwargs: real_client(
            transport=httpx.MockTransport(upstream), **kwargs
        ),
    )
    auth = {"Authorization": "Bearer " + settings.token}

    with TestClient(app) as client:
        app.state.ingest.store.put("viewer", "service", {"state": "READY"})
        ranged = client.get(
            "/bp/job-a/video.mp4?download=1",
            headers={
                **auth,
                "Range": "bytes=2-5",
                "If-Range": '"v1"',
                "Accept": "video/mp4",
                "X-Not-Forwarded": "secret",
            },
        )
        headed = client.head(
            "/_next/static/chunk.js?build=1",
            headers={
                **auth,
                "Range": "bytes=0-3",
                "Accept": "text/javascript",
            },
        )

    assert ranged.status_code == 206
    assert ranged.content == b"2345"
    assert ranged.headers["content-range"] == "bytes 2-5/10"
    assert headed.status_code == 206
    assert headed.content == b""
    assert headed.headers["content-range"] == "bytes 0-3/4"
    assert [(request.method, str(request.url)) for request in seen] == [
        (
            "GET",
            f"http://127.0.0.1:{settings.viewer_ui_port}"
            "/bp/job-a/video.mp4?download=1",
        ),
        (
            "HEAD",
            f"http://127.0.0.1:{settings.viewer_ui_port}"
            "/_next/static/chunk.js?build=1",
        ),
    ]
    assert seen[0].headers["range"] == "bytes=2-5"
    assert seen[0].headers["if-range"] == '"v1"'
    assert seen[0].headers["accept"] == "video/mp4"
    assert seen[1].headers["range"] == "bytes=0-3"
    assert seen[1].headers["accept"] == "text/javascript"
    for request in seen:
        assert "authorization" not in request.headers
        assert "cookie" not in request.headers
        assert "x-not-forwarded" not in request.headers


def test_embedded_data_routes_keep_ranges_and_server_ownership(
    settings_factory, tmp_path, monkeypatch
):
    import httpx

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
    upstream_calls = []

    async def upstream(request):
        upstream_calls.append(request)
        return httpx.Response(200, stream=httpx.ByteStream(b"unexpected"))

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        "pipeline.server.viewer_gateway.httpx.AsyncClient",
        lambda **kwargs: real_client(
            transport=httpx.MockTransport(upstream), **kwargs
        ),
    )

    media = "/_data/bp/job-a/resolve/main/videos/a.mp4"
    with TestClient(app) as client:
        client.cookies.set("pipeline_server", "opaque-user-cookie")
        full = client.get(media)
        ranged = client.get(media, headers={"Range": "bytes=2-5"})
        headed = client.head(media, headers={"Range": "bytes=2-5"})
        multi = client.get(media, headers={"Range": "bytes=0-1,3-4"})
        config = client.get("/api/config")
        asset = client.get("/assets/app.js")
        root = client.get("/")
        uploads = client.get("/api/uploads")
        assert client.get("/unknown-root-path").status_code == 404
        missing_origin = client.post(media)
        same_origin = client.post(
            media, headers={"Origin": "http://testserver"}
        )

    assert full.status_code == 200
    assert full.content == b"0123456789"
    assert ranged.status_code == 206
    assert ranged.content == b"2345"
    assert ranged.headers["content-range"] == "bytes 2-5/10"
    assert headed.status_code == 206
    assert headed.content == b""
    assert headed.headers["content-range"] == "bytes 2-5/10"
    assert multi.status_code == 416
    assert multi.headers["content-range"] == "bytes */10"
    assert config.status_code == 200
    assert config.json()["role"] == "server"
    assert config.json()["current_user"]["username"] == "alice"
    assert asset.status_code == 200
    assert "text/javascript" in asset.headers["content-type"]
    assert root.status_code == 200
    assert "text/html" in root.headers["content-type"]
    assert uploads.status_code == 200
    assert missing_origin.status_code == 403
    assert same_origin.status_code == 405
    assert upstream_calls == []


def test_embedded_data_rejects_unknown_and_cross_dataset_escapes(
    settings_factory, tmp_path
):
    settings = settings_factory(
        "server",
        viewer_gateway={"embedded": True},
    )
    app = create_app(settings, background=False)
    data = make_dataset(tmp_path, "viewer")
    other = make_dataset(tmp_path, "other")
    (tmp_path / "secret.txt").write_text("secret")
    (data / "link").symlink_to(tmp_path / "secret.txt")
    (other / "videos/cross.mp4").symlink_to(data / "videos/a.mp4")
    app.state.viewer.registry.register("job-a", data)
    app.state.viewer.registry.register("job-b", other)
    auth = {"Authorization": "Bearer " + settings.token}

    with TestClient(app, headers=auth) as client:
        paths = (
            "/_data/bp/job-a/resolve/main/../secret.txt",
            "/_data/bp/job-a/resolve/main/%2e%2e/secret.txt",
            "/_data/bp/job-a/resolve/main/link",
            "/_data/bp/job-b/resolve/main/videos/cross.mp4",
            "/_data/bp/job-b/resolve/main/%2e%2e/job-a/videos/a.mp4",
            "/_data/bp/missing/resolve/main/meta/info.json",
        )
        responses = {path: client.get(path) for path in paths}

    for path, response in responses.items():
        assert response.status_code == 404, path


def test_embedded_viewer_serves_startup_page_without_proxy(
    settings_factory, tmp_path, monkeypatch
):
    import httpx

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
    upstream_calls = []

    async def upstream(request):
        upstream_calls.append(request)
        return httpx.Response(200, stream=httpx.ByteStream(b"unexpected"))

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        "pipeline.server.viewer_gateway.httpx.AsyncClient",
        lambda **kwargs: real_client(
            transport=httpx.MockTransport(upstream), **kwargs
        ),
    )
    with TestClient(app) as client:
        client.cookies.set("pipeline_server", "opaque-user-cookie")
        page = client.get(
            "/bp/job-a/episode_0", headers={"Accept": "text/html"}
        )
        data = client.get("/bp/job-a/episode_0")

    assert page.status_code == 200
    assert "正在启动 LeRobot Visualizer" in page.text
    assert data.status_code == 503
    assert upstream_calls == []


def test_server_viewer_transport_is_opt_in(settings_factory):
    settings = settings_factory("server")
    app = create_app(settings, background=False)
    with TestClient(
        app, headers={"Authorization": "Bearer " + settings.token}
    ) as client:
        assert client.get("/bp/job-a/episode_0").status_code == 404
        assert client.get("/api/config").status_code == 200


def test_registry_registers_expires_and_removes(settings_factory, tmp_path):
    s = settings_factory("viewer")
    store = Store(s.root / "state")
    registry = PreviewRegistry(store, ttl=60)
    data = make_dataset(tmp_path)
    row = registry.register("job-1", data)
    assert row["repo_id"] == repo_id_for("job-1") == "bp/job-1"
    assert registry.lookup("bp/job-1")["dataset_path"] == str(data)
    assert registry.lookup("bp/other") is None
    assert registry.lookup("evil/job-1") is None
    assert registry.lookup("bp/../job-1") is None
    again = registry.register("job-1", data)
    assert again["created_at"] == row["created_at"]
    registry.remove("job-1")
    assert registry.lookup("bp/job-1") is None
    registry.remove("job-1")

    registry.register("job-2", data)
    stale = store.get("preview", "job-2")
    stale["expires_at"] = time.time() - 1
    store.put("preview", "job-2", stale)
    assert registry.lookup("bp/job-2") is None
    assert registry.prune() == 1
    assert registry.active() == []


def test_many_datasets_are_served_side_by_side(settings_factory, tmp_path):
    s = settings_factory("viewer")
    store = Store(s.root / "state")
    registry = PreviewRegistry(store, ttl=300)
    first = make_dataset(tmp_path, "one")
    second = make_dataset(tmp_path, "two")
    (second / "videos/a.mp4").write_bytes(b"ABCDEFGHIJ")
    registry.register("job-a", first)
    registry.register("job-b", second)
    store.put("viewer", "service", {"state": "READY"})
    with TestClient(create_gateway(s)) as browser:
        browser.post("/api/login", json={"token": s.token})
        a = browser.get("/_data/bp/job-a/resolve/main/videos/a.mp4")
        b = browser.get("/_data/bp/job-b/resolve/main/videos/a.mp4")
        assert a.content == b"0123456789" and b.content == b"ABCDEFGHIJ"
        ranged = browser.get(
            "/_data/bp/job-a/resolve/main/videos/a.mp4",
            headers={"Range": "bytes=2-5"},
        )
        assert ranged.status_code == 206
        assert ranged.content == b"2345"
        assert ranged.headers["content-range"] == "bytes 2-5/10"
        suffix = browser.get(
            "/_data/bp/job-a/resolve/main/videos/a.mp4",
            headers={"Range": "bytes=-3"},
        )
        assert suffix.status_code == 206 and suffix.content == b"789"
        bad = browser.get(
            "/_data/bp/job-a/resolve/main/videos/a.mp4",
            headers={"Range": "bytes=50-60"},
        )
        assert bad.status_code == 416
        multi = browser.get(
            "/_data/bp/job-a/resolve/main/videos/a.mp4",
            headers={"Range": "bytes=0-1,3-4"},
        )
        assert multi.status_code == 416
        head = browser.head("/_data/bp/job-a/resolve/main/videos/a.mp4")
        assert head.status_code == 200


def test_data_service_confines_paths_and_unknown_repos(settings_factory, tmp_path):
    s = settings_factory("viewer")
    store = Store(s.root / "state")
    registry = PreviewRegistry(store, ttl=300)
    data = make_dataset(tmp_path)
    (tmp_path / "secret.txt").write_text("secret")
    (data / "link").symlink_to(tmp_path / "secret.txt")
    registry.register("job-a", data)
    store.put("viewer", "service", {"state": "READY"})
    with TestClient(create_gateway(s)) as browser:
        browser.post("/api/login", json={"token": s.token})
        for path in (
            "/_data/bp/job-a/resolve/main/../secret.txt",
            "/_data/bp/job-a/resolve/main/%2e%2e/secret.txt",
            "/_data/bp/job-a/resolve/main/link",
            "/_data/bp/job-a/resolve/main/missing.bin",
            "/_data/bp/job-a/resolve/dev/videos/a.mp4",
            "/_data/bp/job-a/videos/a.mp4",
        ):
            assert browser.get(path).status_code == 404, path
        gone = browser.get("/_data/bp/nope/resolve/main/meta/info.json")
        assert gone.status_code == 404
        assert "预览已过期" in gone.json()["detail"]
        # Cookie-authenticated writes require a same-origin Origin header.
        assert browser.post("/_data/bp/job-a/resolve/main/x").status_code == 403
        assert (
            browser.post(
                "/_data/bp/job-a/resolve/main/x",
                headers={"Origin": "http://testserver"},
            ).status_code
            == 405
        )


def test_gateway_proxies_ui_when_service_ready(
    settings_factory, tmp_path, monkeypatch
):
    import httpx

    s = settings_factory("viewer", import_roots=[tmp_path])
    store = Store(s.root / "state")
    seen = []

    async def upstream(request):
        seen.append(request)
        return httpx.Response(200, stream=httpx.ByteStream(b"<html>ui</html>"))

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        "pipeline.server.viewer_gateway.httpx.AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(upstream), **kwargs),
    )
    app = create_gateway(s)
    app.state.viewer_transport.registry.register(
        "job-a", make_recycle_dataset(tmp_path)
    )
    with TestClient(app) as browser:
        browser.post("/api/login", json={"token": s.token})
        waiting = browser.get("/bp/job-a/episode_0", headers={"Accept": "text/html"})
        assert waiting.status_code == 200
        assert 'http-equiv="refresh"' in waiting.text
        assert "正在启动 LeRobot Visualizer" in waiting.text
        assert browser.get("/bp/job-a/episode_0").status_code == 503

        store.put("viewer", "service", {"state": "READY"})
        page = browser.get(
            "/bp/job-a/episode_0",
            headers={"Authorization": "Bearer " + s.token},
        )
        assert page.status_code == 200 and page.content == b"<html>ui</html>"
        assert (
            str(seen[-1].url)
            == f"http://127.0.0.1:{s.viewer_ui_port}/bp/job-a/episode_0"
        )
        assert "cookie" not in seen[-1].headers
        assert "authorization" not in seen[-1].headers
        asset = browser.get("/_next/static/chunk.js")
        assert asset.status_code == 200 and asset.content == b"<html>ui</html>"
        assert (
            str(seen[-1].url)
            == f"http://127.0.0.1:{s.viewer_ui_port}/_next/static/chunk.js"
        )
        assert "cookie" not in seen[-1].headers
        assert "authorization" not in seen[-1].headers


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


def test_root_redirects_to_latest_preview(settings_factory, tmp_path):
    s = settings_factory("viewer")
    store = Store(s.root / "state")
    registry = PreviewRegistry(store, ttl=300)
    store.put("viewer", "service", {"state": "READY"})
    with TestClient(create_gateway(s)) as browser:
        browser.post("/api/login", json={"token": s.token})
        empty = browser.get("/", follow_redirects=False)
        assert empty.status_code == 503
        assert "预览尚未就绪" in empty.json()["detail"]
        data = make_dataset(tmp_path)
        registry.register("job-a", data)
        row = store.get("preview", "job-a")
        row["created_at"] = time.time() - 100
        store.put("preview", "job-a", row)
        registry.register("job-b", data)
        root = browser.get("/", follow_redirects=False)
        assert root.status_code == 307
        assert root.headers["location"] == "/bp/job-b/episode_0"


def test_transport_registers_root_only_when_standalone(
    settings_factory, tmp_path
):
    settings = settings_factory("viewer")
    store = Store(settings.root / "state")
    registry = PreviewRegistry(store, ttl=300)
    registry.register("job-a", make_dataset(tmp_path))
    embedded_app = FastAPI()
    standalone_app = FastAPI()
    transport = ViewerTransport(settings, store, registry)
    try:
        transport.register(embedded_app, embedded=True)
        transport.register(standalone_app, embedded=False)
        with TestClient(embedded_app) as browser:
            assert browser.get("/").status_code == 404
        with TestClient(standalone_app) as browser:
            root = browser.get("/", follow_redirects=False)
            assert root.status_code == 307
            assert root.headers["location"] == "/bp/job-a/episode_0"
    finally:
        transport.close()


def test_gateway_recycle_routes_redirect_and_render_page(
    settings_factory, tmp_path
):
    settings = settings_factory("viewer", import_roots=[tmp_path])
    store = Store(settings.root / "state")
    registry = PreviewRegistry(store, ttl=300)
    registry.register("job-a", make_recycle_dataset(tmp_path))
    store.put("viewer", "service", {"state": "READY"})
    app = create_gateway(settings)
    app.state.viewer_transport.recycle.change("bp/job-a", "trash", [1], 0)
    with TestClient(app) as client:
        client.post("/api/login", json={"token": settings.token})
        redirect = client.get(
            "/bp/job-a/episode_1", follow_redirects=False
        )
        assert redirect.status_code == 303
        assert redirect.headers["location"] == "/bp/job-a/trash"
        page = client.get("/bp/job-a/trash")
        assert page.status_code == 200
        assert "Episode 回收站" in page.text
        assert "永久清理所选" in page.text


def test_doctor_is_computed_per_dataset_and_rendered_light(
    settings_factory, tmp_path, monkeypatch
):
    s = settings_factory("viewer")
    store = Store(s.root / "state")
    registry = PreviewRegistry(store, ttl=300)
    data = make_dataset(tmp_path)
    registry.register("job-a", data)
    registry.register("job-b", make_dataset(tmp_path, "other"))
    calls = []

    class Result:
        returncode = 1
        stderr = ""

        def __init__(self, path):
            self.stdout = json.dumps(
                {
                    "overall_severity": "WARN",
                    "total_episodes": 2,
                    "total_frames": 10,
                    "checks": [
                        {
                            "name": "Action Quality",
                            "severity": "WARN",
                            "messages": [
                                {"severity": "WARN", "message": f"<b>{path}</b>"}
                            ],
                        }
                    ],
                }
            )

    def fake_run(command, **kwargs):
        calls.append(command)
        return Result(command[3])

    monkeypatch.setattr("pipeline.server.viewer_gateway.subprocess.run", fake_run)
    monkeypatch.setattr(
        "pipeline.server.viewer_gateway.executable_path", lambda value: "/bin/letools"
    )
    with TestClient(create_gateway(s)) as browser:
        browser.post("/api/login", json={"token": s.token})
        page = browser.get("/_data/doctor/?dataset=bp/job-a")
        assert page.status_code == 200
        assert "color-scheme: light" in page.text
        assert "&lt;b&gt;" in page.text and "<b>" not in page.text
        assert str(data) in page.text
        again = browser.get("/_data/doctor?dataset=bp/job-a")
        assert again.status_code == 200 and len(calls) == 1
        other = browser.get("/_data/doctor/report.json?dataset=bp/job-b")
        assert other.json()["overall_severity"] == "WARN" and len(calls) == 2
        assert browser.get("/_data/doctor/?dataset=bp/none").status_code == 404
        assert browser.get("/_data/doctor/").status_code == 404


def test_open_registers_without_starting_a_process_per_dataset(
    settings_factory, tmp_path, monkeypatch
):
    s = settings_factory("server")
    store = Store(s.root / "state")
    manager = ViewerManager(s, store)
    data = make_dataset(tmp_path)
    started = []
    monkeypatch.setattr(
        "pipeline.server.viewer.dataset_for_job", lambda settings, job: data
    )
    monkeypatch.setattr(manager, "ensure_service", lambda: started.append(1) or {})
    first = manager.open({"id": "job-1"})
    second = manager.open({"id": "job-2"})
    assert started == [1, 1]
    assert first["url"].endswith("/bp/job-1/episode_0")
    assert second["url"].endswith("/bp/job-2/episode_0")
    status = manager.status()
    assert {p["job_id"] for p in status["previews"]} == {"job-1", "job-2"}
    assert manager.close("job-1")["state"] == "CLOSED"
    assert {p["job_id"] for p in manager.status()["previews"]} == {"job-2"}


def test_service_is_idempotent_and_reports_unavailable(settings_factory, monkeypatch):
    s = settings_factory("server")
    manager = ViewerManager(s, Store(s.root / "state"))
    monkeypatch.setattr("pipeline.server.viewer.executable_path", lambda value: None)
    manager.autostart()
    status = manager.status()
    assert status["state"] == "UNAVAILABLE"
    assert "letools" in status["error"]
    with pytest.raises(DomainError):
        manager.ensure_service()


def test_poll_marks_ready_restarts_and_prunes(settings_factory, monkeypatch):
    import httpx

    s = settings_factory("server")
    store = Store(s.root / "state")
    manager = ViewerManager(s, store)

    class Alive:
        pid = 4_000_000
        returncode = None

        def poll(self):
            return None

    manager.process = Alive()
    store.put("viewer", "service", {"state": "STARTING", "pid": 4_000_000})
    monkeypatch.setattr(
        "pipeline.server.viewer.httpx.get",
        lambda *a, **k: httpx.Response(200),
    )
    manager.poll()
    assert manager.status()["state"] == "READY"

    class Dead(Alive):
        returncode = 3

        def poll(self):
            return 3

    manager.process = Dead()
    manager.poll()
    status = manager.status()
    assert status["state"] == "FAILED" and "退出：3" in status["error"]
    assert manager.process is None

    restarted = []
    monkeypatch.setattr(manager, "autostart", lambda: restarted.append(1))
    row = store.get("viewer", "service")
    row["retry_at"] = time.time() - 1
    store.put("viewer", "service", row)
    manager.poll()
    assert restarted == [1]


def test_recover_reaps_recorded_orphan(settings_factory, monkeypatch):
    s = settings_factory("server")
    store = Store(s.root / "state")
    store.put("viewer", "service", {"state": "READY", "pid": 4_000_000})
    manager = ViewerManager(s, store)
    killed = []
    monkeypatch.setattr(manager, "_kill_group", lambda pid: killed.append(pid))
    monkeypatch.setattr(manager, "_ports_busy", lambda: False)
    manager.recover()
    assert killed == [4_000_000]
    row = store.get("viewer", "service")
    assert row["state"] == "INTERRUPTED" and row["pid"] is None


def test_kill_group_ignores_unrelated_pid(settings_factory):
    s = settings_factory("server")
    manager = ViewerManager(s, Store(s.root / "state"))
    # PID 1 belongs to init and None/garbage must never be signalled.
    manager._kill_group(1)
    manager._kill_group(None)
    manager._kill_group("1234")


def test_server_api_opens_lists_and_closes_previews(
    settings_factory, tmp_path, monkeypatch
):
    s = settings_factory("server")
    app = create_app(s, background=False)
    data = make_dataset(tmp_path)
    monkeypatch.setattr(
        "pipeline.server.viewer.dataset_for_job", lambda settings, job: data
    )
    monkeypatch.setattr(app.state.viewer, "ensure_service", lambda: {"state": "READY"})
    headers = {"Authorization": "Bearer " + s.token}
    store = app.state.ingest.store
    store.put("job", "job-9", {"id": "job-9", "state": "CONVERTED"})
    with TestClient(app) as http:
        opened = http.post("/api/jobs/job-9/viewer", headers=headers)
        assert opened.status_code == 200
        assert opened.json()["url"].endswith("/bp/job-9/episode_0")
        status = http.get("/api/viewer", headers=headers).json()
        assert [p["job_id"] for p in status["previews"]] == ["job-9"]
        closed = http.delete("/api/jobs/job-9/viewer", headers=headers)
        assert closed.json()["state"] == "CLOSED"
        assert http.get("/api/viewer", headers=headers).json()["previews"] == []
        assert (
            http.delete("/api/jobs/missing/viewer", headers=headers).status_code == 404
        )


def test_viewer_patch_is_version_checked_and_idempotent(tmp_path):
    (tmp_path / "source").mkdir()
    (tmp_path / ".letools-source.json").write_text(
        json.dumps({"upstream_commit": UPSTREAM})
    )
    config = tmp_path / "source/next.config.ts"
    config.write_text("const nextConfig: NextConfig = {\n};\n")
    patch_cache(tmp_path)
    once = config.read_text()
    patch_cache(tmp_path)
    assert config.read_text() == once
    assert "env: { DATASET_URL:" in once
    (tmp_path / ".letools-source.json").write_text(
        json.dumps({"upstream_commit": "unexpected"})
    )
    with pytest.raises(RuntimeError):
        patch_cache(tmp_path)


def test_presentation_patch_accepts_an_already_patched_source(tmp_path, monkeypatch):
    from pipeline.server import viewer_bootstrap as bootstrap

    source = tmp_path / "source"
    assets = tmp_path / "assets"
    episode = source / "src/app/[org]/[dataset]/[episode]/episode-viewer.tsx"
    videos = source / "src/components/simple-videos-player.tsx"
    episode.parent.mkdir(parents=True)
    videos.parent.mkdir(parents=True)
    assets.mkdir()
    (source / "src/app/layout.tsx").write_text('<html lang="zh-CN">\n')
    episode.write_text(
        '<div className="viewer-tabbar flex">\n'
        '<div className="viewer-identity">\n'
    )
    videos.write_text(
        "export const SimpleVideosPlayer = ({ videosInfo: sourceVideosInfo }) => {\n"
        '  const className = "camera-stage camera-view";\n'
        "};\n"
    )
    destinations = {
        "data-recharts.tsx": "src/components/data-recharts.tsx",
        "theme-toggle.tsx": "src/components/theme-toggle.tsx",
        "chart-model.ts": "src/utils/chart-model.ts",
        "viewer-ui.css": "src/app/viewer-ui.css",
        "side-nav.tsx": "src/components/side-nav.tsx",
        "episode-scores.tsx": "src/components/episode-scores.tsx",
        "episode-score-filter.tsx": "src/components/episode-score-filter.tsx",
        "episode-score-filter-model.ts": "src/utils/episode-score-filter-model.ts",
        "episode-navigation-state.ts": "src/utils/episode-navigation-state.ts",
        "subtask-review-model.ts": "src/utils/subtask-review-model.ts",
        "annotation-review-client.ts": "src/utils/annotation-review-client.ts",
        "task-review-client.ts": "src/utils/task-review-client.ts",
        "task-annotation-model.ts": "src/utils/task-annotation-model.ts",
        "use-annotation-review.ts": "src/utils/use-annotation-review.ts",
        "instruction-model.ts": "src/utils/instruction-model.ts",
        "instruction-editor.tsx": "src/components/instruction-editor.tsx",
        "instruction-episode-editor.tsx": "src/components/instruction-episode-editor.tsx",
    }
    for name, target in destinations.items():
        (assets / name).write_text(name)
        (source / target).parent.mkdir(parents=True, exist_ok=True)
        (source / target).write_text("old")
    versions = source / "src/utils/versionUtils.ts"
    versions.write_text(
        'import { authHeaders } from "./auth";\n'
        "const CACHE_TTL_MS = 5 * 60 * 1000;\n"
        "function cache(repoId, data) {\n"
        "    datasetInfoCache.set(repoId, {\n"
        "    });\n"
        "}\n"
    )
    (source / "src/utils/parquetUtils.ts").write_text("const parquetFileCache = new Map();\n")

    monkeypatch.setattr(bootstrap, "UI_ASSETS", assets)
    monkeypatch.setattr(bootstrap, "patch_score_filter", lambda cache: None)
    monkeypatch.setattr(bootstrap, "patch_subtask_review", lambda cache: None)
    monkeypatch.setattr(bootstrap, "patch_instruction_editor", lambda cache: None)

    bootstrap.patch_presentation(tmp_path)

    assert "viewer-tabbar" in episode.read_text()
    assert "videosInfo: sourceVideosInfo" in videos.read_text()


def test_instruction_patch_removes_duplicate_navigation_import(tmp_path):
    from pipeline.server.viewer_instructions import MARKER, patch_instruction_editor

    page = tmp_path / "source/src/app/[org]/[dataset]/[episode]/episode-viewer.tsx"
    page.parent.mkdir(parents=True)
    navigation_import = (
        "import {\n"
        "  initialEpisodePage,\n"
        "  pageForEpisode,\n"
        '} from "@/utils/episode-navigation-state";\n'
    )
    page.write_text(
        navigation_import
        + MARKER
        + '\nimport { InstructionEditor } from "@/components/instruction-editor";\n'
        + navigation_import
        + "export const viewer = true;\n"
    )

    patch_instruction_editor(tmp_path)

    patched = page.read_text()
    assert patched.count(navigation_import) == 1
    assert "export const viewer = true;" in patched


def test_build_is_skipped_until_inputs_change(tmp_path, monkeypatch):
    from pipeline.server import viewer_bootstrap as bootstrap

    channel_patch = tmp_path / "channels.patch"
    channel_patch.write_text("layout-v1")
    monkeypatch.setattr(bootstrap, "CHANNEL_PATCH", channel_patch)
    score_filter_patch = tmp_path / "score-filter.patch"
    score_filter_patch.write_text("score-filter-v1")
    monkeypatch.setattr(bootstrap, "SCORE_FILTER_PATCH", score_filter_patch)
    (tmp_path / "source/.next").mkdir(parents=True)
    (tmp_path / ".letools-source.json").write_text(json.dumps({"fingerprint": "f1"}))
    builds = []
    monkeypatch.setattr(
        bootstrap.subprocess, "run", lambda *a, **k: builds.append(a[0])
    )
    bootstrap.build_if_needed(tmp_path, "bun", {}, "http://x/_data")
    bootstrap.build_if_needed(tmp_path, "bun", {}, "http://x/_data")
    assert len(builds) == 1
    bootstrap.build_if_needed(tmp_path, "bun", {}, "http://y/_data")
    assert len(builds) == 2
    (tmp_path / ".letools-source.json").write_text(json.dumps({"fingerprint": "f2"}))
    bootstrap.build_if_needed(tmp_path, "bun", {}, "http://y/_data")
    assert len(builds) == 3
    channel_patch.write_text("layout-v2")
    bootstrap.build_if_needed(tmp_path, "bun", {}, "http://y/_data")
    assert len(builds) == 4
    assets = tmp_path / "ui"
    assets.mkdir()
    (assets / "chart.tsx").write_text("new chart presentation")
    monkeypatch.setattr(bootstrap, "UI_ASSETS", assets)
    bootstrap.build_if_needed(tmp_path, "bun", {}, "http://y/_data")
    assert len(builds) == 5
    (assets / "chart.tsx").write_text("updated chart presentation")
    bootstrap.build_if_needed(tmp_path, "bun", {}, "http://y/_data")
    assert len(builds) == 6
    score_filter_patch.write_text("score-filter-v2")
    bootstrap.build_if_needed(tmp_path, "bun", {}, "http://y/_data")
    assert len(builds) == 7


def test_corrupt_viewer_state_is_not_reported_as_stopped(settings_factory):
    settings = settings_factory("server")
    app = create_app(settings, background=False)
    with TestClient(app) as http:
        path = settings.root / "state/viewer/service.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text("broken JSON")
        response = http.get(
            "/api/viewer", headers={"Authorization": "Bearer " + settings.token}
        )
        assert response.status_code == 500
        assert "状态文件损坏" in response.json()["detail"]
        with pytest.raises(DomainError) as invalid:
            ViewerManager(settings, app.state.ingest.store).recover()
        assert invalid.value.code == 500


def test_gateway_requires_the_server_token(settings_factory, tmp_path):
    s = settings_factory("viewer")
    store = Store(s.root / "state")
    registry = PreviewRegistry(store, ttl=300)
    registry.register("job-a", make_dataset(tmp_path))
    store.put("viewer", "service", {"state": "READY"})
    with TestClient(create_gateway(s)) as browser:
        # Browsers get a login form instead of a JSON error.
        page = browser.get("/bp/job-a/episode_0", headers={"Accept": "text/html"})
        assert page.status_code == 200
        assert "登录数据预览" in page.text
        assert 'id="token"' in page.text
        assert browser.get("/login").status_code == 200
        # Data, video ranges and the Doctor endpoints stay closed.
        for path in (
            "/_data/bp/job-a/resolve/main/meta/info.json",
            "/_data/doctor/?dataset=bp/job-a",
            "/_data/doctor/report.json?dataset=bp/job-a",
        ):
            assert browser.get(path).status_code == 401, path
        assert browser.get("/api/viewer").status_code == 401
        assert browser.post("/api/login", json={"token": "wrong"}).status_code == 401

        assert browser.post("/api/login", json={"token": s.token}).status_code == 200
        assert (
            browser.get("/_data/bp/job-a/resolve/main/meta/info.json").status_code
            == 200
        )
        me = browser.get("/_data/bp/job-a/resolve/main/meta/info.json")
        assert me.status_code == 200

        browser.post("/api/logout")
        assert (
            browser.get("/_data/bp/job-a/resolve/main/meta/info.json").status_code
            == 401
        )


def test_gateway_accepts_bearer_for_automation(settings_factory, tmp_path):
    s = settings_factory("viewer")
    store = Store(s.root / "state")
    registry = PreviewRegistry(store, ttl=300)
    registry.register("job-a", make_dataset(tmp_path))
    store.put("viewer", "service", {"state": "READY"})
    headers = {"Authorization": "Bearer " + s.token}
    with TestClient(create_gateway(s)) as browser:
        ok = browser.get("/_data/bp/job-a/resolve/main/meta/info.json", headers=headers)
        assert ok.status_code == 200
        assert (
            browser.get(
                "/_data/bp/job-a/resolve/main/meta/info.json",
                headers={"Authorization": "Bearer nope"},
            ).status_code
            == 401
        )
