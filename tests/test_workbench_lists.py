import copy
import json
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from test_dataset_management import seed_raw, seed_result
from test_merges import seed

from pipeline.server.app import create_app
from pipeline.server.listings import ReadCache, RecordIndex
from pipeline.shared.core import DomainError, Store


@pytest.fixture
def workbench(settings_factory):
    app = create_app(settings_factory("server"), background=False)
    return app


def test_scoped_pages_search_auth_and_validation(workbench):
    app = workbench
    store = app.state.ingest.store
    for n in range(65):
        store.put(
            "job",
            f"job-{n:03}",
            {
                "id": f"job-{n:03}",
                "state": "FAILED",
                "created_at": n,
                "robot_type": "alpha" if n < 60 else "beta",
                "dataset_name": f"Dataset {n:03}",
                "doctor_report": {"content": "x" * 1000},
            },
        )
    admin = {"Authorization": "Bearer " + app.state.runner.settings.token}
    with TestClient(app) as http:
        url = "/api/listings/jobs?robot_type=alpha"
        assert http.get(url).status_code == 401
        assert (
            http.get(
                url,
                headers={
                    "Authorization": "Bearer " + app.state.runner.settings.device_token
                },
            ).status_code
            == 401
        )
        response = http.get(url, headers=admin).json()
        assert response["total"] == 60
        assert len(response["items"]) == 50
        assert response["items"][0]["id"] == "job-059"
        assert all(
            row["robot_type"] == "alpha" and "doctor_report" not in row
            for row in response["items"]
        )
        second = http.get(url + "&offset=50", headers=admin).json()
        assert len(second["items"]) == 10
        assert not {r["id"] for r in response["items"]} & {
            r["id"] for r in second["items"]
        }
        search = http.get(url + "&q=DATASET%20002", headers=admin).json()
        assert [r["id"] for r in search["items"]] == ["job-002"]
        robots = http.get("/api/listings/robots", headers=admin).json()["robot_types"]
        assert {"alpha", "beta"}.issubset(robots)
        for suffix in (
            "?limit=201&robot_type=alpha",
            "?offset=-1&robot_type=alpha",
            "",
        ):
            assert (
                http.get("/api/listings/jobs" + suffix, headers=admin).status_code
                == 422
            )
        assert (
            http.get(
                "/api/listings/missing?robot_type=alpha", headers=admin
            ).status_code
            == 404
        )


def test_warm_pages_share_metadata_and_do_not_scan_other_robots(workbench, monkeypatch):
    app = workbench
    seed(app)
    seed(app, "other", robot_type="other_arm")
    lists, store = app.state.lists, app.state.ingest.store
    lists.index.warm()
    scans, reads = Counter(), Counter()
    original_all, original_read = store.all, Path.read_text

    def count_all(kind, **kwargs):
        scans[kind] += 1
        return original_all(kind, **kwargs)

    def count_read(path, *args, **kwargs):
        if path.name == "info.json":
            reads[str(path)] += 1
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(store, "all", count_all)
    monkeypatch.setattr(Path, "read_text", count_read)
    for resource in (
        "jobs",
        "merge-candidates",
        "version-conversion-candidates",
        "annotation-candidates",
    ):
        page = lists.page(resource, "test_arm")
        assert page["total"] == len(page["items"]) == 1
        assert page["items"][0]["robot_type"] == "test_arm"
    for _ in range(3):
        lists.status("test_arm")
        lists.page("jobs", "test_arm")
        lists.page("merge-candidates", "test_arm")
    assert not scans
    assert sum(reads.values()) == 1
    assert all("other" not in path for path in reads)
    result = lists.page("jobs", "test_arm")
    result["items"][0]["dataset_name"] = "mutated by consumer"
    assert lists.page("jobs", "test_arm")["items"][0]["dataset_name"] == "first"


def test_local_changes_update_by_id_and_reassign_robot(workbench, monkeypatch):
    store, index = workbench.state.ingest.store, workbench.state.lists.index
    first = {"id": "one", "state": "RUNNING", "robot_type": "alpha"}
    store.put("job", "one", first)
    index.warm()
    alpha_version = index.version("job", "alpha")
    beta_version = index.version("job", "beta")
    monkeypatch.setattr(
        store,
        "all",
        lambda *a, **kw: pytest.fail("warm local write must not scan state"),
    )
    store.put("job", "one", {**first, "robot_type": "beta", "state": "COMPLETED"})
    assert index.select("job", "alpha") == []
    assert len(index.select("job", "beta")) == 1
    assert index.version("job", "alpha") > alpha_version
    assert index.version("job", "beta") > beta_version
    store.delete("job", "one")
    assert index.select("job", "beta") == []


def test_external_writes_and_metadata_reconcile_after_ttl(workbench, monkeypatch):
    app = workbench
    job, path = seed(app)
    lists, store = app.state.lists, app.state.ingest.store
    clock = [100.0]
    monkeypatch.setattr("pipeline.server.listings.time.monotonic", lambda: clock[0])
    assert lists.page("merge-candidates", "test_arm")["items"][0]["fps"] == 10
    external = Store(store.path)
    external.put("job", job["id"], {**job, "dataset_name": "external edit"})
    info = json.loads((path / "meta/info.json").read_text())
    (path / "meta/info.json").write_text(json.dumps({**info, "fps": 20}))
    assert lists.page("merge-candidates", "test_arm")["items"][0]["fps"] == 10
    clock[0] += 31
    refreshed = lists.page("merge-candidates", "test_arm")["items"][0]
    assert refreshed["fps"] == 20
    assert refreshed["dataset_name"] == "external edit"
    # A metadata-only external edit also expires even without a job write.
    (path / "meta/info.json").write_text(json.dumps({**info, "fps": 30}))
    clock[0] += 31
    assert lists.page("merge-candidates", "test_arm")["items"][0]["fps"] == 30


def test_lightweight_dataset_list_validates_only_on_delete(workbench, monkeypatch):
    app = workbench
    job, path = seed_result(app)
    manager, lists = app.state.datasets, app.state.lists
    validations = []
    validate = manager._validate

    def checked(*args, **kwargs):
        validations.append(True)
        return validate(*args, **kwargs)

    monkeypatch.setattr(manager, "_validate", checked)
    assert lists.page("datasets", job["robot_type"])["items"][0]["validation_pending"]
    assert not validations
    assert lists.deletion_preview("result", job["id"])["paths"] == [str(path)]
    assert len(validations) == 1
    # A new active consumer after opening the dialog must block actual deletion.
    app.state.ingest.store.put(
        "job",
        "consumer",
        {
            "id": "consumer",
            "state": "QUEUED",
            "operation": "merge",
            "source_job_ids": [job["id"]],
        },
    )
    headers = {"Authorization": "Bearer " + app.state.runner.settings.token}
    with TestClient(app) as http:
        response = http.post(
            "/api/dataset-deletions",
            headers=headers,
            json={
                "kind": "result",
                "id": job["id"],
                "confirmation": job["dataset_name"],
            },
        )
        assert response.status_code == 409
        assert path.is_dir()
    assert len(validations) >= 2


def test_deleting_raw_groups_and_result_rows_stay_visible(workbench):
    app = workbench
    raw = seed_raw(app)
    result, _ = seed_result(app)
    store, lists = app.state.ingest.store, app.state.lists
    store.put("upload", raw["id"], {**raw, "state": "DELETING"})
    store.put("job", result["id"], {**result, "state": "DELETE_FAILED"})
    assert lists.page("collections", raw["robot_type"])["items"] == []
    rows = lists.page("datasets", raw["robot_type"])["items"]
    assert {row["state"] for row in rows} == {"DELETING", "DELETE_FAILED"}
    assert all(not row["deletable"] for row in rows)


def test_collections_defer_manifest_checks_until_selection(workbench, monkeypatch):
    app = workbench
    first = seed_raw(app)
    second = seed_raw(app, "another-session")
    store, lists = app.state.ingest.store, app.state.lists
    store.put("upload", second["id"], {**second, "collection_id": "another-group"})
    lists.index.warm()
    reads = []
    original = app.state.ingest.manifest

    def manifest(row):
        reads.append(row["id"])
        return original(row)

    monkeypatch.setattr(app.state.ingest, "manifest", manifest)
    page = lists.page("collections", first["robot_type"])
    assert page["total"] == 2
    assert not reads
    assert all(row["conversion_checks_pending"] for row in page["items"])
    detail = lists.collection_detail(first["robot_type"], first["collection_id"])
    assert not detail["conversion_checks_pending"]
    assert "conversion_blockers" in detail["uploads"][0]
    assert set(reads) == {first["id"]}
    lists.collection_detail(first["robot_type"], first["collection_id"])
    assert len(reads) == 1
    with pytest.raises(DomainError, match="数据集不存在"):
        lists.collection_detail("another-robot", first["collection_id"])


def test_cached_candidates_never_bypass_mutation_checks(workbench, monkeypatch):
    app = workbench
    monkeypatch.setattr(
        "pipeline.server.version_conversions.executable_path", lambda _: sys.executable
    )
    job, path = seed(app)
    assert app.state.lists.page("version-conversion-candidates", "test_arm")["items"][
        0
    ]["selectable"]
    (path / "meta/info.json").write_text("{}")
    headers = {"Authorization": "Bearer " + app.state.runner.settings.token}
    with TestClient(app) as http:
        response = http.post(
            "/api/version-conversions",
            headers=headers,
            json={
                "source_job_id": job["id"],
                "target_format": "v2.1",
            },
        )
        assert response.status_code == 409
        assert (
            "元数据" in response.json()["detail"]
            or "LeRobot" in response.json()["detail"]
        )


def test_recycle_without_id_and_inherited_robot_status(workbench):
    app = workbench
    job, _ = seed(app)
    store = app.state.ingest.store
    store.put("recycle", "old-entry", {"entries": []})
    store.put(
        "annotation",
        "annotation-one",
        {"id": "annotation-one", "job_id": job["id"], "state": "RUNNING"},
    )
    store.put(
        "check",
        "check-one",
        {"id": "check-one", "job_id": job["id"], "state": "RUNNING"},
    )
    status = app.state.lists.status("test_arm")
    assert status["active_annotations"] == ["annotation-one"]
    assert status["active_checks"] == ["check-one"]
    assert app.state.lists.index.select("recycle")[0]["id"] == "old-entry"
    assert "id" not in store.all("recycle")[0]
    store.put("recycle", "old-entry", {"entries": [{"episode": 0}]})
    assert (
        app.state.lists.status("test_arm")["revisions"]["recycle"]
        > status["revisions"]["recycle"]
    )
    store.put("job", job["id"], {**job, "robot_type": "other_arm"})
    assert app.state.lists.status("test_arm")["active_annotations"] == []
    assert app.state.lists.status("other_arm")["active_annotations"] == [
        "annotation-one"
    ]


def test_conversion_and_merge_queues_paginate_independently(workbench):
    store, lists = workbench.state.ingest.store, workbench.state.lists
    for operation in ("conversion", "merge"):
        for n in range(4):
            identifier = f"{operation}-{n}"
            store.put(
                "job",
                identifier,
                {
                    "id": identifier,
                    "state": "FAILED",
                    "robot_type": "test",
                    "operation": operation,
                    "created_at": n,
                },
            )
    for resource, operation in (("jobs", "conversion"), ("merge-jobs", "merge")):
        page = lists.page(resource, "test", offset=2, limit=2)
        assert page["total"] == 4
        assert len(page["items"]) == 2
        assert all(row["operation"] == operation for row in page["items"])


def test_legacy_trim_candidates_are_cached_then_scoped(workbench):
    app = workbench
    seed(app)
    seed(app, "other", robot_type="other_arm")
    calls = []

    def candidates():
        calls.append(True)
        return [
            {"id": "first", "robot_type": "test_arm"},
            {"id": "other", "robot_type": "other_arm"},
        ]

    app.state.runner.trims = SimpleNamespace(candidates=candidates)
    for _ in range(2):
        page = app.state.lists.page("trim-candidates", "test_arm")
        assert page["items"] == [{"id": "first", "robot_type": "test_arm"}]
    assert calls == [True]


def test_scoped_candidates_keep_robot_when_old_metadata_omits_it(workbench):
    app = workbench
    seed(app)
    seed(app, "other", robot_type="other_arm")
    seen = []

    def candidates(jobs=None, metadata=None):
        seen.extend(job["id"] for job in jobs)
        return [{"id": job["id"], "robot_type": None} for job in jobs]

    app.state.runner.trims = SimpleNamespace(candidates=candidates)
    page = app.state.lists.page("trim-candidates", "test_arm")
    assert page["items"] == [{"id": "first", "robot_type": "test_arm"}]
    assert seen == ["first"]


def test_inplace_trim_is_not_a_second_dataset_and_delivery_is_readonly(workbench):
    app = workbench
    job, _ = seed(app)
    store = app.state.ingest.store
    store.put(
        "job",
        "trim-alias",
        {**job, "id": "trim-alias", "operation": "manual_trim", "in_place": True},
    )
    for resource in (
        "results",
        "datasets",
        "merge-candidates",
        "version-conversion-candidates",
    ):
        page = app.state.lists.page(resource, "test_arm")
        assert page["total"] == 1
        assert page["items"][0]["id"] == job["id"]
    assert app.state.lists.page("jobs", "test_arm")["total"] == 2
    store.put("job", job["id"], {**job, "immutable_delivery": True})
    row = app.state.lists.page("datasets", "test_arm")["items"][0]
    assert not row["deletable"]
    assert "只读" in row["reason"]


def test_write_notifications_do_not_wait_for_index_scan(tmp_path, monkeypatch):
    store = Store(tmp_path / "state")
    index = RecordIndex(store)
    entered, release = threading.Event(), threading.Event()
    scan = store.all

    def blocked_scan(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return scan(*args, **kwargs)

    monkeypatch.setattr(store, "all", blocked_scan)
    with ThreadPoolExecutor(max_workers=2) as pool:
        reading = pool.submit(index.select, "job")
        assert entered.wait(2)
        try:
            row = {"id": "one", "robot_type": "test", "state": "FAILED"}
            pool.submit(store.put, "job", "one", copy.deepcopy(row)).result(timeout=2)
        finally:
            release.set()
        assert reading.result(timeout=2) == [row]
    assert index.select("job", "test") == [row]


def test_slow_metadata_does_not_block_unrelated_cached_reads():
    cache = ReadCache()
    entered, release = threading.Event(), threading.Event()
    loads = []

    def slow():
        loads.append(True)
        entered.set()
        assert release.wait(5)
        return {"value": "slow"}

    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(cache.get, "slow", slow)
        assert entered.wait(2)
        second = pool.submit(cache.get, "slow", slow)
        try:
            assert (
                pool.submit(cache.get, "fast", lambda: "ready").result(timeout=2)
                == "ready"
            )
        finally:
            release.set()
        a, b = first.result(timeout=2), second.result(timeout=2)
    assert loads == [True]
    a["value"] = "changed"
    assert b["value"] == "slow"
