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
