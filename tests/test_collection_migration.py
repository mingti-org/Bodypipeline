import json
from pathlib import Path

import pytest
from test_collection_batches import seed

from pipeline.server.collection_migration import apply_split, plan_split, rollback_split
from pipeline.server.ingest import IngestService
from pipeline.shared.collections import stored_collection_metadata
from pipeline.shared.core import DomainError, atomic_json, digest, file_hash


def prepare(settings_factory):
    ingest = IngestService(settings_factory("server"))
    rows = [seed(ingest, ingest.settings, "one"), seed(ingest, ingest.settings, "two")]
    assignments = [
        {
            "upload_id": row["id"],
            "session_id": row["session_id"],
            "manifest_hash": row["manifest_hash"],
            "original_collection_id": row["collection_id"],
            "collection_date": day,
        }
        for row, day in zip(rows, ["20261102", "20261113"])
    ]
    return ingest, rows, assignments


def test_split_and_rollback_preserve_files_manifests_and_conversion_lineage(
    settings_factory, tmp_path
):
    ingest, rows, assignments = prepare(settings_factory)
    dataset = tmp_path / "converted/dataset"
    dataset.mkdir(parents=True)
    job = {
        "id": "converted",
        "state": "CONVERTED",
        "collection_id": rows[0]["collection_id"],
        "upload_ids": [r["id"] for r in rows],
        "dataset_path": str(dataset),
        "source_manifest_hashes": {r["id"]: r["manifest_hash"] for r in rows},
        "source_paths": [r["raw_path"] for r in rows],
    }
    ingest.store.put("job", job["id"], job)
    lineage = dataset.parent / "lineage.json"
    original_lineage = {
        "collection_id": rows[0]["collection_id"],
        "sources": [{"upload_id": r["id"], "raw_path": r["raw_path"]} for r in rows],
    }
    atomic_json(lineage, original_lineage)
    sealed = {r["id"]: ingest.manifest(r, fresh=True) for r in rows}
    plan = plan_split(ingest, assignments)
    journal = tmp_path / "journal.json"
    assert apply_split(ingest, plan, journal)["state"] == "COMPLETED"
    migrated = [ingest.store.get("upload", r["id"]) for r in rows]
    assert len({r["collection_id"] for r in migrated}) == 2
    for row in migrated:
        m = ingest.manifest(row, fresh=True)
        assert m == sealed[row["id"]] and digest(m) == row["manifest_hash"]
        for f in m["files"]:
            assert file_hash(Path(row["raw_path"]) / f["path"]) == f["sha256"]
        assert (
            stored_collection_metadata(m, row)["collection_id"] == row["collection_id"]
        )
        assert ingest.create(m, "default", "")["raw_path"] == row["raw_path"]
    updated_job = ingest.store.get("job", job["id"])
    assert updated_job["dataset_path"] == job["dataset_path"]
    assert updated_job["source_manifest_hashes"] == job["source_manifest_hashes"]
    assert len(updated_job["source_collection_ids"]) == 2
    assert updated_job["source_paths"] == [r["raw_path"] for r in migrated]
    assert (
        json.loads(lineage.read_text())["sources"][0]["raw_path"]
        == migrated[0]["raw_path"]
    )
    assert rollback_split(ingest, journal)["state"] == "ROLLED_BACK"
    assert ingest.store.get("job", job["id"]) == job
    assert json.loads(lineage.read_text()) == original_lineage
    for row in rows:
        assert ingest.store.get("upload", row["id"]) == row
        assert Path(row["raw_path"]).is_dir()


def test_split_refuses_partial_collection_active_jobs_and_changed_data(
    settings_factory, tmp_path
):
    ingest, rows, assignments = prepare(settings_factory)
    with pytest.raises(DomainError, match="全部"):
        plan_split(ingest, assignments[:1])
    ingest.store.put(
        "job",
        "running",
        {"id": "running", "state": "RUNNING", "upload_ids": [rows[0]["id"]]},
    )
    with pytest.raises(DomainError, match="尚未结束"):
        plan_split(ingest, assignments)
    ingest.store.put(
        "job",
        "running",
        {"id": "running", "state": "FAILED", "upload_ids": [rows[0]["id"]]},
    )
    plan = plan_split(ingest, assignments)
    changed = dict(rows[0], created_at=0)
    ingest.store.put("upload", changed["id"], changed)
    with pytest.raises(DomainError, match="变化"):
        apply_split(ingest, plan, tmp_path / "journal.json")
    assert not (tmp_path / "journal.json").exists()


def test_failure_between_rename_and_record_update_rolls_back(
    settings_factory, tmp_path, monkeypatch
):
    ingest, rows, assignments = prepare(settings_factory)
    plan = plan_split(ingest, assignments)
    put = ingest.store.put
    failed = False

    def interrupt(kind, key, value):
        nonlocal failed
        if kind == "upload" and value.get("collection_migration") and not failed:
            failed = True
            raise OSError("injected write failure")
        return put(kind, key, value)

    monkeypatch.setattr(ingest.store, "put", interrupt)
    journal = tmp_path / "journal.json"
    with pytest.raises(OSError, match="injected"):
        apply_split(ingest, plan, journal)
    assert json.loads(journal.read_text())["state"] == "ROLLED_BACK"
    for row in rows:
        assert ingest.store.get("upload", row["id"]) == row
        assert Path(row["raw_path"]).is_dir()


def test_derived_jobs_record_recursive_source_collections(settings_factory):
    ingest, rows, assignments = prepare(settings_factory)
    direct = {
        "id": "direct",
        "state": "CONVERTED",
        "upload_ids": [rows[0]["id"]],
        "dataset_path": "/tmp/direct/dataset",
    }
    derived = {
        "id": "derived",
        "state": "CONVERTED",
        "operation": "merge",
        "source_job_ids": [direct["id"]],
        "dataset_path": "/tmp/derived/dataset",
    }
    ingest.store.put("job", direct["id"], direct)
    ingest.store.put("job", derived["id"], derived)
    plan = plan_split(ingest, assignments)
    updates = {u["after"]["id"]: u["after"] for u in plan["state_updates"]}
    assert len(updates["direct"]["source_collection_ids"]) == 1
    assert updates["derived"]["source_collection_ids"] == updates["direct"][
        "source_collection_ids"
    ]


