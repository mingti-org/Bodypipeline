"""End-to-end instruction apply job and Viewer gateway proxy."""

import json
import subprocess
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from pipeline.server.app import create_app
from pipeline.server.instruction_review import (
    ApplyInstructionsRequest,
    InstructionReview,
)
from pipeline.server.viewer import create_gateway
from pipeline.server.viewer_data import PreviewRegistry
from pipeline.shared.identity import Principal

SCENARIO = Path(__file__).parent / "fixtures/merge_scenario.py"


def admin(app):
    return {"Authorization": "Bearer " + app.state.runner.settings.token}


def instructions_template():
    return {
        "parts": [
            {"type": "text", "text": "请使用"},
            {"type": "variable", "key": "object"},
            {"type": "text", "text": "放入"},
            {"type": "variable", "key": "target"},
        ],
        "variables": {
            "object": {
                "label": "物体",
                "options": ["蓝色圆柱"],
                "default": "蓝色圆柱",
            },
            "target": {
                "label": "目标位置",
                "options": ["右侧盒子"],
                "default": "右侧盒子",
            },
        },
    }


@pytest.fixture
def instruction_app(settings_factory, monkeypatch):
    settings = settings_factory("server")
    app = create_app(settings, background=False)
    store = app.state.runner.store
    path = settings.root / "source"
    (path / "meta").mkdir(parents=True)
    (path / "meta/info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "fps": 10,
                "total_episodes": 1,
                "total_frames": 6,
                "robot_type": "test_arm",
                "features": {},
            }
        )
    )
    store.put(
        "job",
        "source",
        {
            "id": "source",
            "state": "CONVERTED",
            "dataset_path": str(path),
            "dataset_name": "fixture",
            "created_at": 1,
            "robot_type": "test_arm",
        },
    )
    PreviewRegistry(store, settings.viewer_ttl).register("source", path)
    monkeypatch.setattr(
        InstructionReview,
        "_source",
        lambda *args: {
            "fps": 10,
            "tasks": [{"task_index": 0, "task": "pick"}],
            "episodes": [
                {
                    "episode_index": 0,
                    "frames": 6,
                    "task_index": 0,
                    "instruction": "pick",
                }
            ],
        },
    )
    return app


def test_instruction_apply_records_submitter(instruction_app, monkeypatch):
    runner = instruction_app.state.runner
    client = TestClient(
        instruction_app,
        headers={"Authorization": "Bearer " + runner.settings.token},
    )
    head = client.get("/api/instructions/source").json()
    head = client.put(
        "/api/instructions/source/template",
        json={
            "revision": head["revision"],
            "source_signature": head["source_signature"],
            "template": instructions_template(),
        },
    ).json()
    head = client.put(
        "/api/instructions/source/episodes/0",
        json={
            "revision": head["revision"],
            "source_signature": head["source_signature"],
            "selections": {
                "object": "蓝色圆柱",
                "target": "右侧盒子",
            },
        },
    ).json()
    principal = Principal(
        username="alice",
        uid=1001,
        login_node="H800-node11",
        execution_id="execution-a",
        auth_method="h800-ssh",
    )
    preflight = []
    monkeypatch.setattr(
        runner,
        "require_principal_path_access",
        lambda active, path, writable: preflight.append(
            (active, path, writable)
        ),
    )

    queued = instruction_app.state.instruction_apply.enqueue(
        "source",
        ApplyInstructionsRequest(
            revision=head["revision"],
            source_signature=head["source_signature"],
        ),
        principal=principal,
    )
    row = runner.store.get("instruction_apply", queued["id"])

    assert preflight == [
        (principal, runner.settings.root / "source", True)
    ]
    assert row["submitted_as"] == "alice"
    assert row["slurm_user"] == "alice"
    assert row["slurm_execution_id"] == "execution-a"
    assert "slurm_execution_id" not in queued


def test_instruction_apply_fingerprint_is_per_submitter(
    instruction_app, monkeypatch
):
    runner = instruction_app.state.runner
    client = TestClient(
        instruction_app,
        headers={"Authorization": "Bearer " + runner.settings.token},
    )
    head = client.get("/api/instructions/source").json()
    head = client.put(
        "/api/instructions/source/template",
        json={
            "revision": head["revision"],
            "source_signature": head["source_signature"],
            "template": instructions_template(),
        },
    ).json()
    head = client.put(
        "/api/instructions/source/episodes/0",
        json={
            "revision": head["revision"],
            "source_signature": head["source_signature"],
            "selections": {
                "object": "蓝色圆柱",
                "target": "右侧盒子",
            },
        },
    ).json()
    body = ApplyInstructionsRequest(
        revision=head["revision"],
        source_signature=head["source_signature"],
    )
    alice = Principal(
        username="alice",
        uid=1001,
        login_node="H800-node11",
        execution_id="execution-a",
        auth_method="h800-ssh",
    )
    bob = Principal(
        username="bob",
        uid=1002,
        login_node="H800-node11",
        execution_id="execution-b",
        auth_method="h800-ssh",
    )
    monkeypatch.setattr(
        runner,
        "require_principal_path_access",
        lambda *args, **kwargs: None,
    )

    first = instruction_app.state.instruction_apply.enqueue(
        "source", body, principal=alice
    )
    active = runner.store.get("instruction_apply", first["id"])
    active["state"] = "FAILED"
    runner.store.put("instruction_apply", active["id"], active)
    second = instruction_app.state.instruction_apply.enqueue(
        "source", body, principal=bob
    )

    first_row = runner.store.get("instruction_apply", first["id"])
    second_row = runner.store.get("instruction_apply", second["id"])
    assert first_row["fingerprint"] != second_row["fingerprint"]
    assert first_row["submitted_as"] == "alice"
    assert second_row["submitted_as"] == "bob"


def test_instruction_apply_routes_pass_the_authenticated_principal(
    instruction_app, monkeypatch
):
    worker = instruction_app.state.instruction_apply
    instruction_app.state.runner.settings.viewer_execution_identity = {
        "enabled": True
    }
    calls = []

    def enqueue(dataset_id, body, episode_indices=None, *, principal):
        calls.append((dataset_id, body, episode_indices, principal))
        return {"id": "queued"}

    monkeypatch.setattr(worker, "enqueue", enqueue)
    client = TestClient(
        instruction_app,
        headers={
            "Authorization": "Bearer " + instruction_app.state.runner.settings.token
        },
    )
    body = {"revision": 0, "source_signature": "a" * 64}

    whole = client.post("/api/instructions/source/apply", json=body)
    episode = client.post(
        "/api/instructions/source/episodes/0/apply", json=body
    )

    assert whole.status_code == 202
    assert episode.status_code == 202
    assert calls[0][0] == "source"
    assert calls[0][2] is None
    assert calls[1][0] == "source"
    assert calls[1][2] == {0}
    assert all(call[3].is_admin for call in calls)


def test_embedded_instruction_apply_passes_browser_principal(
    settings_factory, monkeypatch
):
    settings = settings_factory(
        "server",
        viewer_gateway={"embedded": True},
        viewer_execution_identity={"enabled": True},
        user_sessions={
            "enabled": True,
            "login_nodes": ["H800-node11"],
        },
    )
    app = create_app(settings, background=False)
    user = Principal(
        username="alice",
        uid=1001,
        login_node="H800-node11",
        execution_id="execution-a",
        auth_method="h800-ssh",
    )
    monkeypatch.setattr(
        app.state.user_sessions,
        "principal",
        lambda cookie: user if cookie == "opaque-user-cookie" else None,
    )
    captured = []

    def enqueue(dataset_id, body, episode_indices=None, *, principal):
        captured.append(
            {
                "dataset_id": dataset_id,
                "episode_indices": episode_indices,
                "principal": principal,
            }
        )
        return {"id": "apply-a", "state": "QUEUED"}

    monkeypatch.setattr(app.state.instruction_apply, "enqueue", enqueue)
    with TestClient(app) as client:
        client.cookies.set("pipeline_server", "opaque-user-cookie")
        response = client.post(
            "/api/instructions/source/apply",
            json={
                "revision": 0,
                "source_signature": "a" * 64,
            },
            headers={"Origin": "http://testserver"},
        )

    assert response.status_code == 202
    assert captured[0]["dataset_id"] == "source"
    assert captured[0]["episode_indices"] is None
    assert captured[0]["principal"] is user


@pytest.mark.parametrize("identity_enabled", [True, False])
def test_other_viewer_compute_routes_respect_identity_gate(
    settings_factory, tmp_path, monkeypatch, identity_enabled
):
    settings = settings_factory(
        "server",
        viewer_gateway={"embedded": True},
        viewer_execution_identity={"enabled": identity_enabled},
        user_sessions={
            "enabled": True,
            "login_nodes": ["H800-node11"],
        },
    )
    app = create_app(settings, background=False)
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/info.json").write_text("{}")
    app.state.runner.store.put(
        "job",
        "source",
        {
            "id": "source",
            "state": "CONVERTED",
            "dataset_path": str(dataset),
            "created_at": 1,
        },
    )
    app.state.viewer.registry.register("source", dataset)
    user = Principal(
        username="alice",
        uid=1001,
        login_node="H800-node11",
        execution_id="execution-a",
        auth_method="h800-ssh",
    )
    monkeypatch.setattr(
        app.state.user_sessions,
        "principal",
        lambda cookie: user if cookie == "opaque-user-cookie" else None,
    )
    seen = {}

    def episode_enqueue(
        dataset_id, body, episode_indices=None, *, principal
    ):
        seen["episode"] = (dataset_id, episode_indices, principal)
        return {"id": "apply-a", "state": "QUEUED"}

    def manual_enqueue(
        job_id,
        episode,
        ranges,
        name=None,
        run_doctor=False,
        in_place=True,
        *,
        principal,
    ):
        seen["manual"] = (job_id, principal)
        return {
            "id": "trim-a",
            "state": "QUEUED",
            "operation": "manual_trim",
            "dataset_path": str(dataset),
        }

    def review_enqueue(dataset_id, body, *, principal):
        seen["review"] = (dataset_id, principal)
        return {"id": "export-a", "state": "QUEUED"}

    def score_start(preview, *, principal):
        seen["score"] = (preview["job_id"], principal)
        return {"state": "QUEUED"}

    def score_status(preview, *, principal):
        seen["score_status"] = (preview["job_id"], principal)
        return {"state": "NONE"}

    monkeypatch.setattr(
        app.state.instruction_apply, "enqueue", episode_enqueue
    )
    monkeypatch.setattr(app.state.trims, "enqueue_manual", manual_enqueue)
    monkeypatch.setattr(
        app.state.runner.subtask_exports,
        "enqueue_review",
        review_enqueue,
    )
    monkeypatch.setattr(
        app.state.viewer_transport.scores, "start", score_start
    )
    monkeypatch.setattr(
        app.state.viewer_transport.scores, "status", score_status
    )
    origin = {"Origin": "http://testserver"}

    with TestClient(app) as client:
        client.cookies.set("pipeline_server", "opaque-user-cookie")
        episode = client.post(
            "/api/instructions/source/episodes/0/apply",
            json={"revision": 0, "source_signature": "a" * 64},
            headers=origin,
        )
        manual = client.post(
            "/api/manual-trims/source",
            json={"episode_index": 0, "remove_ranges": [[1, 2]]},
            headers=origin,
        )
        review = client.post(
            "/api/subtask-review/source/exports",
            json={"revision": 0, "source_signature": "a" * 64},
            headers=origin,
        )
        score = client.post(
            "/api/episode-scores/source",
            json={
                "username": "mallory",
                "execution_id": "untrusted-handle",
                "is_admin": True,
            },
            headers=origin,
        )
        score_status_response = client.get(
            "/api/episode-scores/source",
            headers=origin,
        )
        client.cookies.clear()
        anonymous_score = client.post(
            "/api/episode-scores/source",
            json={
                "username": "mallory",
                "execution_id": "untrusted-handle",
                "is_admin": True,
            },
            headers=origin,
        )

    assert episode.status_code == 202
    assert manual.status_code == 202
    assert review.status_code == 202
    assert score.status_code == 202
    assert score_status_response.status_code == 200
    assert anonymous_score.status_code == 401
    assert seen["episode"][0] == "source"
    assert seen["episode"][1] == {0}
    assert seen["manual"][0] == "source"
    assert seen["review"][0] == "source"
    assert seen["score"][0] == "source"
    assert seen["score_status"][0] == "source"
    expected = user if identity_enabled else None
    assert seen["episode"][2] is expected
    assert seen["manual"][1] is expected
    assert seen["review"][1] is expected
    assert seen["score"][1] is expected
    assert seen["score_status"][1] is expected


def test_embedded_compute_apis_keep_machine_token_as_service_account(
    settings_factory, monkeypatch
):
    settings = settings_factory(
        "server",
        viewer_gateway={"embedded": True},
        viewer_execution_identity={"enabled": True},
    )
    app = create_app(settings, background=False)
    captured = []

    def enqueue(dataset_id, body, episode_indices=None, *, principal):
        captured.append(principal)
        return {"id": "apply-a", "state": "QUEUED"}

    monkeypatch.setattr(app.state.instruction_apply, "enqueue", enqueue)
    with TestClient(app) as client:
        response = client.post(
            "/api/instructions/source/apply",
            json={"revision": 0, "source_signature": "a" * 64},
            headers={
                "Authorization": "Bearer "
                + app.state.runner.settings.token
            },
        )

    assert response.status_code == 202
    assert captured[0].is_admin is True
    assert captured[0].username == "service-account"
    assert captured[0].auth_method == "service-token"
    assert (
        app.state.runner.submission_metadata(captured[0])["submitted_as"]
        == "service-account"
    )


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        (
            "/api/instructions/source/apply",
            {"revision": 0, "source_signature": "a" * 64},
        ),
        (
            "/api/instructions/source/episodes/0/apply",
            {"revision": 0, "source_signature": "a" * 64},
        ),
        (
            "/api/manual-trims/source",
            {"episode_index": 0, "remove_ranges": [[1, 2]]},
        ),
        (
            "/api/subtask-review/source/exports",
            {"revision": 0, "source_signature": "a" * 64},
        ),
    ],
)
def test_viewer_compute_bodies_reject_identity_fields(
    settings_factory, monkeypatch, path, payload
):
    settings = settings_factory(
        "server",
        viewer_gateway={"embedded": True},
        user_sessions={
            "enabled": True,
            "login_nodes": ["H800-node11"],
        },
    )
    app = create_app(settings, background=False)
    principal = Principal(
        username="alice",
        uid=1001,
        login_node="H800-node11",
        execution_id="execution-a",
        auth_method="h800-ssh",
    )
    monkeypatch.setattr(
        app.state.user_sessions,
        "principal",
        lambda cookie: principal
        if cookie == "opaque-user-cookie"
        else None,
    )

    def reject_compute(*args, **kwargs):
        pytest.fail("identity-bearing request reached a compute handler")

    monkeypatch.setattr(
        app.state.instruction_apply, "enqueue", reject_compute
    )
    monkeypatch.setattr(app.state.trims, "enqueue_manual", reject_compute)
    monkeypatch.setattr(
        app.state.runner.subtask_exports,
        "enqueue_review",
        reject_compute,
    )
    injected = {
        **payload,
        "username": "mallory",
        "uid": 9999,
        "login_node": "untrusted-node",
        "execution_id": "untrusted-handle",
        "is_admin": True,
    }

    with TestClient(app) as client:
        anonymous = client.post(
            path,
            json=injected,
            headers={"Origin": "http://testserver"},
        )
        client.cookies.set("pipeline_server", "opaque-user-cookie")
        authenticated = client.post(
            path,
            json=injected,
            headers={"Origin": "http://testserver"},
        )

    assert anonymous.status_code == 401
    assert authenticated.status_code == 422


def test_apply_job_rewrites_real_dataset_in_place_and_updates_viewer(
    settings_factory, tmp_path, letools_binary
):
    python = str(Path(letools_binary).with_name("python"))
    fixtures = tmp_path / "fixtures"
    subprocess.run(
        [python, str(SCENARIO), "create", str(fixtures), "v3.0"], check=True
    )
    dataset = fixtures / "input-0"
    dataset_path = str(dataset.resolve())
    settings = settings_factory(
        "server", letools_bin=letools_binary, import_roots=[fixtures]
    )
    app = create_app(settings, background=False)
    runner = app.state.runner
    runner.store.put(
        "job",
        "source",
        {
            "id": "source",
            "state": "CONVERTED",
            "dataset_path": str(dataset),
            "dataset_name": "fixture",
            "created_at": 1,
            "robot_type": "test_arm",
        },
    )
    PreviewRegistry(runner.store, settings.viewer_ttl).register("source", dataset)
    client = TestClient(app, headers=admin(app))

    head = client.get("/api/instructions/source").json()
    saved = client.put(
        "/api/instructions/source/template",
        json={
            "revision": head["revision"],
            "source_signature": head["source_signature"],
            "template": instructions_template(),
        },
    )
    assert saved.status_code == 200, saved.text
    head = saved.json()
    saved = client.put(
        "/api/instructions/source/episodes/0",
        json={
            "revision": head["revision"],
            "source_signature": head["source_signature"],
            "selections": {
                "object": "蓝色圆柱",
                "target": "右侧盒子",
            },
        },
    )
    assert saved.status_code == 200, saved.text
    head = saved.json()
    request = {
        "revision": head["revision"],
        "source_signature": head["source_signature"],
    }
    first = client.post("/api/instructions/source/apply", json=request)
    duplicate = client.post("/api/instructions/source/apply", json=request)
    assert first.status_code == 202, first.text
    assert duplicate.status_code == 202, duplicate.text
    assert duplicate.json()["id"] == first.json()["id"]

    app.state.instruction_apply.tick()
    status = client.get(
        f"/api/instructions/source/apply/{first.json()['id']}"
    ).json()
    assert status["state"] == "CONVERTED", status
    assert status["report"]["updated_episodes"] == [0]
    assert str(dataset.resolve()) == dataset_path

    refreshed = client.get("/api/instructions/source").json()
    assert refreshed["source_signature"] != head["source_signature"]
    episode = client.get("/api/instructions/source/episodes/0").json()
    assert episode["current_instruction"] == "请使用蓝色圆柱放入右侧盒子"
    assert episode["has_draft"] is True
    info = json.loads((dataset / "meta/info.json").read_text())
    assert info["bodypipeline_revision"]

    validation = subprocess.run(
        [letools_binary, "validate", "--deep", str(dataset)],
        capture_output=True,
        text=True,
    )
    assert validation.returncode == 0, validation.stdout + validation.stderr


def test_gateway_proxies_only_registered_instruction_identities(settings_factory, monkeypatch):
    settings = settings_factory("viewer")
    # The gateway validates against the same persisted preview registry.
    from pipeline.server.viewer_data import PreviewRegistry
    from pipeline.shared.core import Store

    store = Store(settings.root / "state")
    store.put("viewer", "service", {"state": "READY"})
    PreviewRegistry(store, settings.viewer_ttl).register(
        "source", settings.root / "dataset"
    )
    (settings.root / "dataset/meta").mkdir(parents=True)
    (settings.root / "dataset/meta/info.json").write_text("{}")
    gateway = create_gateway(settings)
    real_client = httpx.AsyncClient
    calls = []

    def make_client(**kwargs):
        async def response(request):
            calls.append(request)
            assert request.headers["authorization"] == "Bearer " + settings.token
            return httpx.Response(200, json={"revision": 4})

        return real_client(transport=httpx.MockTransport(response))

    monkeypatch.setattr("pipeline.server.viewer.httpx.AsyncClient", make_client)
    with TestClient(gateway) as client:
        headers = {"Authorization": "Bearer " + settings.token}
        assert client.get("/api/instructions/source", headers=headers).json() == {
            "revision": 4
        }
        assert (
            client.get("/api/instructions/missing", headers=headers).status_code
            == 404
        )
        assert len(calls) == 1
        assert str(calls[0].url).endswith("/api/instructions/source")
