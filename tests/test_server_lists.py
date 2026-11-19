import copy
import gzip
import hashlib

from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from test_ingest import manifest

from pipeline.server.app import create_app


def test_collection_summary_preserves_selection_and_runtime_checks(settings_factory):
    s = settings_factory("server")
    app = create_app(s, background=False)
    ingest = app.state.ingest
    m, data = manifest(s)
    first = ingest.create(m, "default", "")
    ingest.chunk(first["id"], 0, 0, hashlib.sha256(data).hexdigest(), data)
    ingest.complete(first["id"])
    second = ingest.create({**m, "session_id": "still-uploading"}, "default", "")
    # Inline manifests are used by imports and must never travel in list polls.
    inline = copy.deepcopy(m)
    inline["session_id"] = "imported-episode"
    inline["provenance"] = {"original_metadata": "x" * 100_000}
    imported = ingest.register_import(
        inline, s.root / "source/manifest.json", s.root / "source", inline=True
    )
    headers = {"Authorization": "Bearer " + s.token}
    with TestClient(app) as http:
        assert http.get("/api/collections?summary=true").status_code == 401
        assert (
            http.get(
                "/api/collections?summary=true",
                headers={"Authorization": "Bearer " + s.device_token},
            ).status_code
            == 401
        )
        full_response = http.get("/api/collections", headers=headers)
        slim_response = http.get("/api/collections?summary=true", headers=headers)
        full = full_response.json()
        slim = slim_response.json()
        assert len(slim_response.content) < len(full_response.content) / 10
        for group, summary in zip(full, slim, strict=True):
            assert {k: v for k, v in group.items() if k != "uploads"} == {
                k: v for k, v in summary.items() if k != "uploads"
            }
            for original, episode in zip(group["uploads"], summary["uploads"], strict=True):
                for key in (
                    "id", "state", "robot_id", "robot_type", "runtime_id",
                    "total_bytes", "file_count", "created_at", "conversion_blockers",
                ):
                    assert episode.get(key) == original.get(key)
                assert "manifest_inline" not in episode
                assert "raw_path" not in episode
        episodes = [u for group in slim for u in group["uploads"]]
        assert {u["id"] for u in episodes} == {first["id"], imported["id"]}
        assert second["id"] not in {u["id"] for u in episodes}
        assert next(u for u in episodes if u["id"] == imported["id"])["import_path"]
        # Existing upload clients and the full collection API retain the manifest.
        uploads = http.get("/api/uploads", headers=headers).json()
        assert next(u for u in uploads if u["id"] == imported["id"])["manifest_inline"] == inline


def test_workbench_collections_publish_only_after_complete(settings_factory):
    s = settings_factory("server")
    app = create_app(s, background=False)
    m, data = manifest(s)
    human = {"Authorization": "Bearer " + s.token}
    device = {"Authorization": "Bearer " + s.device_token}
    robot_type = m["profile"]["robot_type"]

    def collection_snapshot(http):
        legacy = http.get("/api/collections", headers=human).json()
        listing = http.get(
            f"/api/listings/collections?robot_type={robot_type}", headers=human
        ).json()
        return legacy, listing["items"], listing["total"]

    with TestClient(app) as http:
        created = http.post(
            "/device/uploads",
            headers=device,
            json={"manifest": m, "target": "default", "prefix": ""},
        ).json()
        uid = created["id"]
        first = data[:5]
        assert (
            http.put(
                f"/device/uploads/{uid}/files/0?offset=0",
                headers={**device, "x-chunk-sha256": hashlib.sha256(first).hexdigest()},
                content=first,
            ).status_code
            == 200
        )

        progress = http.get(f"/device/uploads/{uid}", headers=device).json()
        assert progress["state"] == "UPLOADING"
        assert progress["offsets"] == [len(first)]
        assert collection_snapshot(http) == ([], [], 0)

    # A process restart, including a warm index, must not publish an
    # incomplete row from persisted state.
    restarted = create_app(s, background=False)
    restarted.state.lists.index.warm()
    with TestClient(restarted) as http:
        assert collection_snapshot(http) == ([], [], 0)

        rest = data[len(first) :]
        assert (
            http.put(
                f"/device/uploads/{uid}/files/0?offset={len(first)}",
                headers={**device, "x-chunk-sha256": hashlib.sha256(rest).hexdigest()},
                content=rest,
            ).status_code
            == 200
        )
        assert http.post(f"/device/uploads/{uid}/complete", headers=device).json()[
            "state"
        ] == "COMMITTED"
        legacy, listing, total = collection_snapshot(http)
        assert len(legacy) == len(listing) == total == 1
        assert legacy[0]["episode_count"] == legacy[0]["ready_count"] == 1
        assert listing[0]["episode_count"] == 1

        second_m = {**m, "session_id": "session-test-002"}
        second = http.post(
            "/device/uploads",
            headers=device,
            json={"manifest": second_m, "target": "default", "prefix": ""},
        ).json()
        second_first = data[:4]
        http.put(
            f"/device/uploads/{second['id']}/files/0?offset=0",
            headers={
                **device,
                "x-chunk-sha256": hashlib.sha256(second_first).hexdigest(),
            },
            content=second_first,
        )
        assert collection_snapshot(http) == (legacy, listing, total)
        assert http.get(
            f"/api/listings/status?robot_type={robot_type}", headers=human
        ).json()["counts"]["uploads"] == 1

        second_rest = data[len(second_first) :]
        http.put(
            f"/device/uploads/{second['id']}/files/0?offset={len(second_first)}",
            headers={
                **device,
                "x-chunk-sha256": hashlib.sha256(second_rest).hexdigest(),
            },
            content=second_rest,
        )
        for _ in range(2):
            assert http.post(
                f"/device/uploads/{second['id']}/complete", headers=device
            ).json()["state"] == "COMMITTED"
        legacy, listing, total = collection_snapshot(http)
        assert len(legacy) == len(listing) == total == 1
        assert legacy[0]["episode_count"] == legacy[0]["ready_count"] == 2
        assert listing[0]["episode_count"] == 2
        assert http.get(
            f"/api/listings/status?robot_type={robot_type}", headers=human
        ).json()["counts"]["uploads"] == 2


def test_collection_summary_keeps_manifest_failures_visible(settings_factory, monkeypatch):
    s = settings_factory("server")
    app = create_app(s, background=False)
    ingest = app.state.ingest
    m, data = manifest(s)
    row = ingest.create(m, "default", "")
    ingest.chunk(row["id"], 0, 0, hashlib.sha256(data).hexdigest(), data)
    ingest.complete(row["id"])

    def unavailable(*args, **kwargs):
        raise OSError("source unavailable")

    monkeypatch.setattr(ingest, "manifest", unavailable)
    with TestClient(app) as http:
        response = http.get(
            "/api/collections?summary=true", headers={"Authorization": "Bearer " + s.token}
        )
        assert response.status_code == 200
        episode = response.json()[0]["uploads"][0]
        assert episode["id"] == row["id"]
        assert episode["conversion_blockers"][0]["field"] == "manifest"


def test_job_summary_keeps_reports_in_detail(settings_factory):
    s = settings_factory("server")
    app = create_app(s, background=False)
    row = {
        "id": "large-job", "state": "FAILED", "created_at": 1,
        "profile": {"robot_type": "example", "large_metadata": "x" * 100_000},
        "doctor_report": {"checks": [{"messages": "x" * 100_000}]},
        "trim_report": {"episodes": ["x" * 100_000]},
        "upload_ids": [f"episode-{n}" for n in range(1000)],
        "selection_report": {"input_count": 1000, "accepted_count": 999, "skipped_count": 1},
        "source_job_ids": ["source-one", "source-two"],
        "error": "conversion failed",
    }
    app.state.ingest.store.put("job", row["id"], row)
    headers = {"Authorization": "Bearer " + s.token}
    with TestClient(app) as http:
        response = http.get("/api/jobs?summary=true", headers=headers)
        summary = response.json()[0]
        assert len(response.content) < 2000
        assert summary["robot_type"] == "example"
        for field in ("selection_report", "source_job_ids", "error"):
            assert summary[field] == row[field]
        for path in ("/api/jobs", "/api/jobs/large-job"):
            full = http.get(path, headers=headers).json()
            if isinstance(full, list):
                full = full[0]
            for field in ("profile", "doctor_report", "trim_report", "upload_ids"):
                assert field not in summary
                assert full[field] == row[field]


def test_delivery_batch_name_is_projected_for_display(settings_factory):
    s = settings_factory("server")
    app = create_app(s, background=False)
    delivery_path = s.root / "deliveries/sim/task/batch-20261111-000005/lerobot"
    delivery = {
        "id": "delivery-job",
        "state": "CONVERTED",
        "created_at": 2,
        "robot_type": "sim",
        "dataset_name": "sim-sarm-drawer-cycle",
        "dataset_path": str(delivery_path),
        "dataset_ready": True,
        "immutable_delivery": True,
        "delivery_id": "sim-drawer-cycle-20261111-000005",
    }
    regular = {
        "id": "regular-job",
        "state": "CONVERTED",
        "created_at": 1,
        "robot_type": "sim",
        "dataset_name": "regular-dataset",
        "dataset_ready": True,
    }
    app.state.ingest.store.put("job", delivery["id"], delivery)
    app.state.ingest.store.put("job", regular["id"], regular)
    headers = {"Authorization": "Bearer " + s.token}
    with TestClient(app) as http:
        jobs = http.get("/api/jobs", headers=headers).json()
        delivery_job = next(row for row in jobs if row["id"] == delivery["id"])
        regular_job = next(row for row in jobs if row["id"] == regular["id"])
        datasets = http.get(
            "/api/listings/datasets?robot_type=sim&q=batch-20261111-000005",
            headers=headers,
        ).json()

    assert delivery_job["dataset_name"] == "sim-sarm-drawer-cycle"
    assert delivery_job["batch_name"] == "batch-20261111-000005"
    assert "batch_name" not in regular_job
    assert datasets["total"] == 1
    assert datasets["items"][0]["id"] == delivery["id"]
    assert datasets["items"][0]["dataset_name"] == "sim-sarm-drawer-cycle"
    assert datasets["items"][0]["batch_name"] == "batch-20261111-000005"


def test_list_compression_negotiation_does_not_touch_device_or_media(settings_factory):
    s = settings_factory("server")
    app = create_app(s, background=False)
    for n in range(20):
        app.state.ingest.store.put(
            "job", str(n),
            {"id": str(n), "state": "FAILED", "created_at": n, "error": "reason " * 100},
        )

    @app.get("/api/test-stream")
    def media():
        return StreamingResponse(iter([b"frame" * 1000]), media_type="multipart/x-mixed-replace")

    headers = {"Authorization": "Bearer " + s.token}
    with TestClient(app) as http:
        compressed = http.get("/api/jobs?summary=true", headers={**headers, "Accept-Encoding": "gzip"})
        plain = http.get("/api/jobs?summary=true", headers={**headers, "Accept-Encoding": "identity"})
        assert compressed.headers["content-encoding"] == "gzip"
        assert "accept-encoding" in compressed.headers["vary"].lower()
        assert "content-encoding" not in plain.headers
        assert compressed.json() == plain.json()
        with http.stream(
            "GET", "/api/jobs?summary=true", headers={**headers, "Accept-Encoding": "gzip"}
        ) as response:
            wire = b"".join(response.iter_raw())
        assert gzip.decompress(wire) == plain.content
        assert len(wire) < len(plain.content) / 5
        stream = http.get("/api/test-stream", headers={**headers, "Accept-Encoding": "gzip"})
        assert stream.content == b"frame" * 1000
        assert "content-encoding" not in stream.headers
        targets = http.get(
            "/device/targets",
            headers={"Authorization": "Bearer " + s.device_token, "Accept-Encoding": "gzip"},
        )
        assert targets.status_code == 200
        assert "content-encoding" not in targets.headers
