import copy
import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_ingest import manifest

from pipeline.server.app import create_app
from pipeline.server.ingest import IngestService
from pipeline.server.jobs import JobRunner
from pipeline.shared.collections import (
    collection_metadata,
    dated_batch,
    stored_collection_metadata,
)
from pipeline.shared.core import DomainError, digest


def seed(ingest, settings, session, name="fold cloth", complete=True, **changes):
    m, data = manifest(settings)
    m.update(session_id=session, dataset_name=name, **changes)
    row = ingest.create(m, "default", "")
    if complete:
        ingest.chunk(row["id"], 0, 0, hashlib.sha256(data).hexdigest(), data)
        row = ingest.complete(row["id"])
    return row


def test_legacy_identity_stays_exact_and_batches_are_idempotent(settings_factory):
    settings = settings_factory("server")
    ingest = IngestService(settings)
    m, _ = manifest(settings)
    old = collection_metadata(m, "default", "")
    scope = {
        "name": m["instruction"],
        "project_id": m["project_id"],
        "robot_type": m["profile"]["robot_type"],
        "profile_hash": m["profile_hash"],
        "capture_mode": m["capture_mode"],
        "raw_format": m["raw_format"],
        "target": "default",
        "prefix": "",
    }
    assert old["collection_id"] == digest(scope)[:32]

    one = seed(ingest, settings, "one", **dated_batch("20261102"))
    two = seed(ingest, settings, "two", **dated_batch("20261113"))
    three = seed(ingest, settings, "three", **dated_batch("20261113"))
    assert one["collection_id"] != two["collection_id"] == three["collection_id"]
    assert ingest.create(ingest.manifest(two), "default", "")["id"] == two["id"]
    status = ingest.offsets(two["id"])
    assert status["collection_date"] == "20261113"
    assert status["batch_id"] == "date-20261113"
    assert status["session_id"] == two["session_id"]
    assert status["raw_path"] == two["raw_path"]

    runner = JobRunner(settings, ingest)
    job = runner.enqueue_collection(two["collection_id"], "Insert the screw.")
    assert set(job["upload_ids"]) == {two["id"], three["id"]}
    assert job["collection_date"] == "20261113"
    with pytest.raises(DomainError, match="同一数据集"):
        runner.enqueue_collection(
            two["collection_id"],
            "Insert the screw.",
            upload_ids=[one["id"], two["id"]],
        )


@pytest.mark.parametrize(
    "fields",
    [
        {"collection_date": "20260230", "batch_id": "date-20260230"},
        {"collection_date": "20261113"},
        {"batch_id": "date-20261113"},
        {"collection_date": "20261113", "batch_id": "date-20261102"},
        {"collection_date": 20261113, "batch_id": "date-20261113"},
    ],
)
def test_invalid_batch_rejected_before_upload_is_created(settings_factory, fields):
    ingest = IngestService(settings_factory("server"))
    m, _ = manifest(ingest.settings)
    with pytest.raises(DomainError):
        ingest.create({**m, **fields}, "default", "")
    assert ingest.store.all("upload") == []


def test_device_protocol_exposes_batch_fields(settings_factory):
    settings = settings_factory("server")
    app = create_app(settings, background=False)
    headers = {"Authorization": "Bearer " + settings.device_token}
    m, data = manifest(settings)
    m.update(dataset_name="insert_the_screw-wxz", **dated_batch("20261113"))
    with TestClient(app) as http:
        targets = http.get("/device/targets", headers=headers).json()
        assert targets["collection_batch_version"] == 1
        created = http.post(
            "/device/uploads",
            headers=headers,
            json={"manifest": m, "target": "default", "prefix": ""},
        ).json()
        assert created["collection_date"] == "20261113"
        assert created["batch_id"] == "date-20261113"
        assert created["dataset_name"] == m["dataset_name"]
        uid = created["id"]
        http.put(
            f"/device/uploads/{uid}/files/0?offset=0",
            headers={
                **headers,
                "X-Chunk-SHA256": hashlib.sha256(data).hexdigest(),
            },
            content=data,
        )
        completed = http.post(f"/device/uploads/{uid}/complete", headers=headers).json()
        assert completed["state"] == "COMMITTED"
        assert completed["collection_id"] == created["collection_id"]
        assert completed["collection_date"] == "20261113"
        assert completed["batch_id"] == "date-20261113"
        assert completed["session_id"] == m["session_id"]
        assert Path(completed["raw_path"]).is_dir()
        status = http.get(f"/device/uploads/{uid}", headers=headers).json()
        for key in (
            "collection_id",
            "dataset_name",
            "collection_date",
            "batch_id",
            "session_id",
            "raw_path",
        ):
            assert status[key] == completed[key]


def test_migrated_sealed_manifest_keeps_hash_and_uses_new_batch(settings_factory):
    settings = settings_factory("server")
    app = create_app(settings, background=False)
    ingest = app.state.ingest
    original = seed(ingest, settings, "legacy")
    sealed = ingest.manifest(original)
    migrated = {
        **original,
        **collection_metadata({**sealed, **dated_batch("20261113")}, "default", ""),
        "collection_migration": {
            "version": 1,
            "original_collection_id": original["collection_id"],
            "manifest_hash": original["manifest_hash"],
        },
    }
    old_dir = Path(original["raw_path"]).parent
    new_dir = ingest.final_dir(migrated)
    new_dir.parent.mkdir(parents=True, exist_ok=True)
    old_dir.rename(new_dir)
    migrated["raw_path"] = str(new_dir / "data")
    ingest.store.put("upload", migrated["id"], migrated)

    assert digest(ingest.manifest(migrated, fresh=True)) == original["manifest_hash"]
    assert (
        ingest.create(sealed, "default", "")["collection_id"]
        == migrated["collection_id"]
    )
    with TestClient(app) as http:
        groups = http.get(
            "/api/collections",
            headers={"Authorization": "Bearer " + settings.token},
        ).json()
    assert len(groups) == 1 and groups[0]["id"] == migrated["collection_id"]
    assert groups[0]["collection_date"] == "20261113"
    job = JobRunner(settings, ingest).enqueue_collection(
        migrated["collection_id"], "Insert the screw."
    )
    assert job["upload_ids"] == [original["id"]]
    bad = copy.deepcopy(migrated)
    bad["collection_migration"]["manifest_hash"] = "0" * 64
    with pytest.raises(DomainError):
        stored_collection_metadata(sealed, bad)
