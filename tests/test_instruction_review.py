"""Server drafts for structured per-Episode language instructions."""

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from pipeline.server.app import create_app
from pipeline.server.instruction_review import InstructionReview
from pipeline.server.merges import source_snapshot
from pipeline.server.recycle import dataset_lock
from pipeline.server.viewer_data import PreviewRegistry


def instruction_template():
    return {
        "parts": [
            {"type": "text", "text": "请将"},
            {"type": "variable", "key": "object"},
            {"type": "text", "text": "放入"},
            {"type": "variable", "key": "target"},
        ],
        "variables": {
            "object": {
                "label": "物体",
                "options": ["红色方块", "蓝色圆柱"],
                "default": "红色方块",
            },
            "target": {
                "label": "目标位置",
                "options": ["左侧盒子", "右侧盒子"],
                "default": "左侧盒子",
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
                "total_episodes": 2,
                "total_frames": 11,
                "robot_type": "test_arm",
                "features": {
                    "observation.state": {
                        "dtype": "float32",
                        "shape": [2],
                        "names": ["joint0", "joint1"],
                        "units": ["rad", "normalized"],
                    }
                },
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
            "tasks": [
                {"task_index": 0, "task": "pick"},
                {"task_index": 1, "task": "place"},
            ],
            "episodes": [
                {
                    "episode_index": 0,
                    "frames": 6,
                    "task_index": 0,
                    "instruction": "pick",
                },
                {
                    "episode_index": 1,
                    "frames": 5,
                    "task_index": 1,
                    "instruction": "place",
                },
            ],
        },
    )
    return app


def client_for(app):
    return TestClient(
        app,
        headers={"Authorization": "Bearer " + app.state.runner.settings.token},
    )


def versioned(payload):
    return {
        "revision": payload["revision"],
        "source_signature": payload["source_signature"],
    }


def test_initial_template_episode_selections_and_restart_persistence(instruction_app):
    client = client_for(instruction_app)
    head = client.get("/api/instructions/source").json()
    assert head["revision"] == 0
    assert head["template"]["variables"]["instruction"]["options"] == ["pick", "place"]
    assert [item["current_instruction"] for item in head["episodes"]] == ["pick", "place"]

    saved = client.put(
        "/api/instructions/source/template",
        json={**versioned(head), "template": instruction_template()},
    )
    assert saved.status_code == 200, saved.text
    saved = saved.json()
    assert saved["revision"] == 1

    episode = client.put(
        "/api/instructions/source/episodes/0",
        json={
            **versioned(saved),
            "selections": {
                "object": "蓝色圆柱",
                "target": "右侧盒子",
            },
        },
    )
    assert episode.status_code == 200, episode.text
    episode = episode.json()
    assert episode["rendered"] == "请将蓝色圆柱放入右侧盒子"
    assert episode["revision"] == 2

    restarted = create_app(instruction_app.state.runner.settings, background=False)
    loaded = client_for(restarted).get("/api/instructions/source/episodes/0").json()
    assert loaded["selections"] == {
        "object": "蓝色圆柱",
        "target": "右侧盒子",
    }
    assert loaded["rendered"] == "请将蓝色圆柱放入右侧盒子"
    assert loaded["has_draft"] is True
    store = restarted.state.runner.store
    assert len(store.all("instruction_revision")) == 2


def test_template_changes_report_removed_values_without_destroying_assignments(instruction_app):
    client = client_for(instruction_app)
    head = client.get("/api/instructions/source").json()
    head = client.put(
        "/api/instructions/source/template",
        json={**versioned(head), "template": instruction_template()},
    ).json()
    head = client.put(
        "/api/instructions/source/episodes/0",
        json={
            **versioned(head),
            "selections": {"object": "蓝色圆柱", "target": "左侧盒子"},
        },
    ).json()
    changed = instruction_template()
    changed["variables"]["object"]["options"] = ["红色方块"]
    changed["variables"]["object"]["default"] = "红色方块"
    result = client.put(
        "/api/instructions/source/template",
        json={**versioned(head), "template": changed},
    )
    assert result.status_code == 200, result.text
    assert result.json()["warnings"][0]["episode_index"] == 0
    episode = client.get("/api/instructions/source/episodes/0").json()
    assert episode["selections"]["object"] == "蓝色圆柱"
    assert episode["valid"] is False
    assert "已移除选项" in episode["error"]


def test_in_place_manual_trim_rebases_saved_instruction_draft(instruction_app):
    runner = instruction_app.state.runner
    client = client_for(instruction_app)
    path = runner.settings.root / "source"
    head = client.get("/api/instructions/source").json()
    head = client.put(
        "/api/instructions/source/template",
        json={**versioned(head), "template": instruction_template()},
    ).json()
    head = client.put(
        "/api/instructions/source/episodes/0",
        json={
            **versioned(head),
            "selections": {
                "object": "蓝色圆柱",
                "target": "右侧盒子",
            },
        },
    ).json()

    trim_id = "trim-in-place"
    (path / "meta/bodypipeline-manual-trim.json").write_text(
        json.dumps({"job_id": trim_id})
    )
    published = source_snapshot(path)
    source = runner.store.get("job", "source")
    source.update(
        published_dataset_snapshot=published,
        manual_trim_history=[
            {
                "job_id": trim_id,
                "episode_index": 0,
                "frames_kept": 5,
                "at": 2,
            }
        ],
    )
    runner.store.put("job", "source", source)

    loaded = client.get("/api/instructions/source")
    assert loaded.status_code == 200, loaded.text
    loaded = loaded.json()
    assert loaded["revision"] == head["revision"] + 1
    assert loaded["source_signature"] == published["signature"]
    assert loaded["template"] == instruction_template()
    episode = client.get("/api/instructions/source/episodes/0").json()
    assert episode["selections"] == {
        "object": "蓝色圆柱",
        "target": "右侧盒子",
    }
    revisions = runner.store.all("instruction_revision")
    assert any(
        row["revision"] == head["revision"]
        and row["source_signature"] == head["source_signature"]
        for row in revisions
    )
    assert any(
        row["revision"] == loaded["revision"]
        and row["source_signature"] == published["signature"]
        for row in revisions
    )


def test_chained_manual_trims_rebase_the_latest_instruction_revision(instruction_app):
    runner = instruction_app.state.runner
    client = client_for(instruction_app)
    path = runner.settings.root / "source"
    head = client.get("/api/instructions/source").json()
    head = client.put(
        "/api/instructions/source/template",
        json={**versioned(head), "template": instruction_template()},
    ).json()
    head = client.put(
        "/api/instructions/source/episodes/1",
        json={
            **versioned(head),
            "selections": {
                "object": "红色方块",
                "target": "左侧盒子",
            },
        },
    ).json()

    source = runner.store.get("job", "source")
    history = []
    for index in range(2):
        (path / f"meta/manual-trim-{index}.json").write_text("{}")
        published = source_snapshot(path)
        history.append(
            {
                "job_id": f"trim-{index}",
                "episode_index": index,
                "frames_kept": 5 - index,
                "at": index + 2,
            }
        )
        source.update(
            published_dataset_snapshot=published,
            manual_trim_history=history,
        )
        runner.store.put("job", "source", source)
        loaded = client.get("/api/instructions/source")
        assert loaded.status_code == 200, loaded.text
        head = loaded.json()
        assert head["source_signature"] == published["signature"]

    episode = client.get("/api/instructions/source/episodes/1").json()
    assert episode["selections"] == {
        "object": "红色方块",
        "target": "左侧盒子",
    }
    assert head["revision"] == 4


@pytest.mark.parametrize("signed_cleanup", [True, False])
def test_recycle_cleanup_remaps_saved_instruction_draft(
    instruction_app, monkeypatch, signed_cleanup
):
    runner = instruction_app.state.runner
    client = client_for(instruction_app)
    path = runner.settings.root / "source"
    cleanup_marker = path / "meta/recycle-cleanup.json"

    def source(*_args):
        indexes = (0, 1, 2) if not cleanup_marker.exists() else (0, 1)
        return {
            "fps": 10,
            "tasks": [
                {"task_index": 0, "task": "pick"},
                {"task_index": 1, "task": "place"},
                {"task_index": 2, "task": "stack"},
            ],
            "episodes": [
                {
                    "episode_index": index,
                    "frames": 6 - index,
                    "task_index": index,
                    "instruction": ("pick", "place", "stack")[index],
                }
                for index in indexes
            ],
        }

    monkeypatch.setattr(InstructionReview, "_source", source)
    head = client.get("/api/instructions/source").json()
    head = client.put(
        "/api/instructions/source/template",
        json={**versioned(head), "template": instruction_template()},
    ).json()
    selections = [
        {"object": "红色方块", "target": "左侧盒子"},
        {"object": "蓝色圆柱", "target": "左侧盒子"},
        {"object": "蓝色圆柱", "target": "右侧盒子"},
    ]
    for episode, selected in enumerate(selections):
        response = client.put(
            f"/api/instructions/source/episodes/{episode}",
            json={**versioned(head), "selections": selected},
        )
        assert response.status_code == 200, response.text
        head = response.json()

    before_signature = source_snapshot(path)["signature"]
    cleanup_marker.write_text(json.dumps({"cleanup": "remove-episode-1"}))
    after_signature = source_snapshot(path)["signature"]
    review = InstructionReview(runner.settings, runner.store)
    key, resolved_path, _ = review.resolve("source")
    assert resolved_path == path
    draft = runner.store.get("instruction_draft", key)
    history = {
        "id": "remove-episode-1",
        "from": 0,
        "to": 1,
        "mapping": {"0": 0, "2": 1},
    }
    if signed_cleanup:
        history.update(
            from_signature=before_signature,
            to_signature=after_signature,
        )
    runner.store.put(
        "recycle",
        key,
        {
            "id": key,
            "dataset_path": str(path),
            "entries": {},
            "revision": 1,
            "created_at": 1,
            "cleanup_generation": 1,
            "cleanup_history": [history],
            "operation": {
                "id": "remove-episode-1",
                "state": "completed",
                "mapping": {"0": 0, "2": 1},
                "started_at": draft["updated_at"] + 1,
            },
        },
    )

    loaded = client.get("/api/instructions/source")
    assert loaded.status_code == 200, loaded.text
    loaded = loaded.json()
    assert loaded["revision"] == head["revision"] + 1
    assert loaded["source_signature"] == after_signature
    assert loaded["assignment_count"] == 2

    first = client.get("/api/instructions/source/episodes/0").json()
    assert first["selections"] == selections[0]
    remapped = client.get("/api/instructions/source/episodes/1").json()
    assert remapped["selections"] == selections[2]
    assert remapped["rendered"] == "请将蓝色圆柱放入右侧盒子"

    rebased = runner.store.get("instruction_draft", key)
    assert rebased["last_recycle_rebase"]["cleanup_id"] == "remove-episode-1"
    assert rebased["last_recycle_rebase"]["removed_episodes"] == [1]
    assert rebased["last_recycle_rebase"]["remapped_episodes"] == [[2, 1]]

    stable = client.get("/api/instructions/source")
    assert stable.status_code == 200, stable.text
    assert stable.json()["revision"] == loaded["revision"]


def test_stale_drafts_source_changes_and_locks_are_rejected(instruction_app):
    runner = instruction_app.state.runner
    client = client_for(instruction_app)
    head = client.get("/api/instructions/source").json()
    first = client.put(
        "/api/instructions/source/template",
        json={**versioned(head), "template": instruction_template()},
    )
    assert first.status_code == 200
    stale = client.put(
        "/api/instructions/source/template",
        json={**versioned(head), "template": instruction_template()},
    )
    assert stale.status_code == 409
    assert "其他窗口" in stale.json()["detail"]

    review = InstructionReview(runner.settings, runner.store)
    key, _, _ = review.resolve("source")
    with dataset_lock(runner.store, key):
        assert (
            client.put(
                "/api/instructions/source/template",
                json={**versioned(first.json()), "template": instruction_template()},
            ).status_code
            == 409
        )

    (runner.settings.root / "source/changed").write_text("new")
    assert client.get("/api/instructions/source").status_code == 409


def test_concurrent_instruction_reads_do_not_report_dataset_as_processing(
    instruction_app, monkeypatch
):
    first_read_entered = threading.Event()
    release_first_read = threading.Event()
    call_lock = threading.Lock()
    fixture_source = InstructionReview._source
    call_count = 0

    def slow_first_source(*args):
        nonlocal call_count
        with call_lock:
            call_count += 1
            current_call = call_count
        if current_call == 1:
            first_read_entered.set()
            assert release_first_read.wait(2)
        return fixture_source(*args)

    monkeypatch.setattr(InstructionReview, "_source", slow_first_source)
    first_client = client_for(instruction_app)
    second_client = client_for(instruction_app)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(first_client.get, "/api/instructions/source")
        assert first_read_entered.wait(2)
        second = executor.submit(
            second_client.get, "/api/instructions/source/episodes/0"
        )
        try:
            second_response = second.result(timeout=2)
        finally:
            release_first_read.set()
        first_response = first.result(timeout=2)

    assert first_response.status_code == 200, first_response.text
    assert second_response.status_code == 200, second_response.text


def test_api_validation_episode_bounds_and_dataset_identity(instruction_app, tmp_path):
    client = client_for(instruction_app)
    assert TestClient(instruction_app).get("/api/instructions/source").status_code == 401
    assert client.get("/api/instructions/missing").status_code == 404
    head = client.get("/api/instructions/source").json()
    assert client.get("/api/instructions/source/episodes/99").status_code == 404
    assert (
        client.put(
            "/api/instructions/source/episodes/0",
            json={
                **versioned(head),
                "selections": {"object": "not-an-option", "target": "左侧盒子"},
            },
        ).status_code
        == 422
    )
    assert (
        client.put(
            "/api/instructions/source/template",
            json={**versioned(head), "template": {"parts": [], "variables": {}}},
        ).status_code
        == 422
    )

    store = instruction_app.state.runner.store
    outside = tmp_path / "outside"
    (outside / "meta").mkdir(parents=True)
    (outside / "meta/info.json").write_text("{}")
    PreviewRegistry(store, 60).register("source", outside)
    assert client.get("/api/instructions/source").status_code == 403


def test_active_dataset_job_blocks_apply_and_restart_requeues_running_apply(
    instruction_app,
):
    runner = instruction_app.state.runner
    client = client_for(instruction_app)
    head = client.get("/api/instructions/source").json()
    head = client.put(
        "/api/instructions/source/template",
        json={**versioned(head), "template": instruction_template()},
    ).json()
    head = client.put(
        "/api/instructions/source/episodes/0",
        json={
            **versioned(head),
            "selections": {
                "object": "红色方块",
                "target": "左侧盒子",
            },
        },
    ).json()
    request = versioned(head)

    runner.store.put(
        "job",
        "busy-work",
        {
            "id": "busy-work",
            "state": "RUNNING",
            "dataset_path": str(runner.settings.root / "source"),
            "created_at": 2,
        },
    )
    blocked = client.post("/api/instructions/source/apply", json=request)
    assert blocked.status_code == 409
    assert "运行中的工作台任务" in blocked.json()["detail"]
    runner.store.delete("job", "busy-work")

    runner.store.put(
        "job",
        "busy-trim",
        {
            "id": "busy-trim",
            "state": "RUNNING",
            "operation": "manual_trim",
            "in_place": True,
            "source_path": str(runner.settings.root / "source"),
            "created_at": 2,
        },
    )
    blocked = client.post("/api/instructions/source/apply", json=request)
    assert blocked.status_code == 409
    assert "运行中的工作台任务" in blocked.json()["detail"]
    runner.store.delete("job", "busy-trim")

    queued = client.post("/api/instructions/source/apply", json=request).json()
    running = runner.store.get("instruction_apply", queued["id"])
    running.update(state="RUNNING", stage="写入中")
    runner.store.put("instruction_apply", running["id"], running)
    instruction_app.state.instruction_apply.recover()
    recovered = runner.store.get("instruction_apply", queued["id"])
    assert recovered["state"] == "QUEUED"
    assert "恢复中断任务" in recovered["stage"]


def test_active_instruction_apply_blocks_manual_trim_enqueue(instruction_app):
    runner = instruction_app.state.runner
    client = client_for(instruction_app)
    head = client.get("/api/instructions/source").json()
    head = client.put(
        "/api/instructions/source/template",
        json={**versioned(head), "template": instruction_template()},
    ).json()
    head = client.put(
        "/api/instructions/source/episodes/0",
        json={
            **versioned(head),
            "selections": {
                "object": "红色方块",
                "target": "左侧盒子",
            },
        },
    ).json()
    queued = client.post("/api/instructions/source/apply", json=versioned(head))
    assert queued.status_code == 202, queued.text

    response = client.post(
        "/api/jobs/source/manual-trims",
        json={"episode_index": 0, "remove_ranges": [[1, 2]]},
    )
    assert response.status_code == 409
    assert "语言指令" in response.json()["detail"]
    assert not any(
        row.get("operation") == "manual_trim"
        for row in runner.store.all("job")
    )

    apply_row = runner.store.get("instruction_apply", queued.json()["id"])
    apply_row.update(state="CONVERTED")
    runner.store.put("instruction_apply", apply_row["id"], apply_row)
    runner.settings.letools_bin = sys.executable
    response = client.post(
        "/api/jobs/source/manual-trims",
        json={"episode_index": 0, "remove_ranges": [[1, 2]]},
    )
    assert response.status_code == 202, response.text
    assert response.json()["operation"] == "manual_trim"
    assert response.json()["submitted_as"] == "service-account"
