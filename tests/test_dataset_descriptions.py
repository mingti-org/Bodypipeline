from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from fastapi.testclient import TestClient
from test_dataset_management import seed_result

from pipeline.server.app import create_app
from pipeline.server.dataset_descriptions import (
    DatasetDescriptionRequest,
    DatasetDescriptions,
)
from pipeline.shared.core import DomainError


@pytest.mark.parametrize("robot_type", ["ARX_X5", "g1_brainco", "other_robot"])
def test_description_is_persistent_independent_metadata(settings_factory, robot_type):
    settings = settings_factory("server")
    app = create_app(settings, background=False)
    job, path = seed_result(app, robot_type=robot_type)
    original_info = (path / "meta/info.json").read_bytes()
    url = f"/api/jobs/{job['id']}/description"
    headers = {"Authorization": "Bearer " + settings.token}
    text = "搬运箱子\n两种光照；仅包含成功示教。\n<script>alert('plain text')</script>"
    with TestClient(app) as http:
        assert http.get(url, headers=headers).json() == {
            "id": job["id"],
            "description": "",
            "revision": 0,
            "updated_at": None,
        }
        saved = http.put(
            url, headers=headers, json={"description": text, "revision": 0}
        )
        assert saved.status_code == 200, saved.text
        assert saved.json()["description"] == text
        assert saved.json()["revision"] == 1
        assert saved.json()["updated_at"] > 0
        assert app.state.ingest.store.get("job", job["id"]) == job
        assert (path / "meta/info.json").read_bytes() == original_info
        assert list(path.iterdir()) == [path / "meta"]
    # A new application reads the persisted introduction without a job migration.
    with TestClient(create_app(settings, background=False)) as http:
        assert http.get(url, headers=headers).json() == saved.json()
        cleared = http.put(
            url, headers=headers, json={"description": "", "revision": 1}
        )
        assert cleared.status_code == 200
        assert cleared.json()["description"] == ""
        assert cleared.json()["revision"] == 2


def test_description_auth_validation_and_conflict(settings_factory):
    settings = settings_factory("server")
    app = create_app(settings, background=False)
    job, _ = seed_result(app)
    url = f"/api/jobs/{job['id']}/description"
    headers = {"Authorization": "Bearer " + settings.token}
    body = {"description": "first edit", "revision": 0}
    with TestClient(app) as http:
        for credentials in ({}, {"Authorization": "Bearer " + settings.device_token}):
            assert http.get(url, headers=credentials).status_code == 401
            assert http.put(url, headers=credentials, json=body).status_code == 401
        for invalid in (
            {"description": "x"},
            {**body, "description": "x" * 10001},
            {**body, "description": 1},
            {**body, "description": None},
            {**body, "revision": True},
            {**body, "revision": -1},
            {**body, "revision": "0"},
            {**body, "path": "/tmp/arbitrary-file"},
        ):
            assert http.put(url, headers=headers, json=invalid).status_code == 422
        assert http.put(url, headers=headers, json=body).status_code == 200
        conflict = http.put(
            url, headers=headers, json={**body, "description": "stale browser"}
        )
        assert conflict.status_code == 409
        assert "其他用户" in conflict.json()["detail"]
        assert http.get(url, headers=headers).json()["description"] == "first edit"
        # Cookie sessions keep the existing cross-origin write protection.
        assert (
            http.post("/api/login", json={"token": settings.token}).status_code == 200
        )
        assert http.put(url, json={**body, "revision": 1}).status_code == 403


@pytest.mark.parametrize("state", ["RUNNING", "DELETING", "DELETE_FAILED", "DELETED"])
def test_unavailable_results_cannot_receive_a_description(settings_factory, state):
    settings = settings_factory("server")
    app = create_app(settings, background=False)
    job, _ = seed_result(app, state=state, dataset_ready=False)
    headers = {"Authorization": "Bearer " + settings.token}
    with TestClient(app) as http:
        for jid, expected in ((job["id"], 409), ("missing", 404)):
            url = f"/api/jobs/{jid}/description"
            assert http.get(url, headers=headers).status_code == expected
            assert (
                http.put(
                    url, headers=headers, json={"description": "text", "revision": 0}
                ).status_code
                == expected
            )
    assert app.state.ingest.store.all("dataset_description") == []


def test_simultaneous_edits_do_not_overwrite_each_other(settings_factory):
    settings = settings_factory("server")
    app = create_app(settings, background=False)
    job, _ = seed_result(app)
    service = DatasetDescriptions(settings, app.state.runner)
    barrier = Barrier(2)

    def save(text):
        barrier.wait(timeout=5)
        try:
            return service.save(
                job["id"], DatasetDescriptionRequest(description=text, revision=0)
            )
        except DomainError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(save, ["first", "second"]))
    assert sum(isinstance(value, dict) for value in results) == 1
    assert results.count(409) == 1
    assert service.get(job["id"])["revision"] == 1


def test_result_list_previews_refresh_without_changing_job_state(settings_factory):
    settings = settings_factory("server")
    app = create_app(settings, background=False)
    job, _ = seed_result(app)
    seed_result(app, "other", robot_type="another_robot")
    headers = {"Authorization": "Bearer " + settings.token}
    list_url = "/api/listings/results?robot_type=agilex_aloha"
    status_url = "/api/listings/status?robot_type=agilex_aloha"
    description_url = f"/api/jobs/{job['id']}/description"
    with TestClient(app) as http:
        empty = http.get(list_url, headers=headers).json()
        assert empty["items"][0]["dataset_description_preview"] == ""
        before = http.get(status_url, headers=headers).json()["revisions"]
        other_before = http.get(
            "/api/listings/status?robot_type=another_robot", headers=headers
        ).json()["revisions"]
        text = "搬运箱子\n<script>alert('plain text')</script>\n" + "详细介绍" * 1000
        assert (
            http.put(
                description_url,
                headers=headers,
                json={"description": text, "revision": 0},
            ).status_code
            == 200
        )
        fresh = http.get(list_url, headers=headers).json()
        preview = fresh["items"][0]["dataset_description_preview"]
        assert preview == text[:300].rstrip() + "…"
        assert len(preview) == 301
        assert fresh["total"] == 1
        after = http.get(status_url, headers=headers).json()["revisions"]
        assert after["dataset_description"] > before["dataset_description"]
        assert after["job"] == before["job"]
        other_after = http.get(
            "/api/listings/status?robot_type=another_robot", headers=headers
        ).json()["revisions"]
        assert other_after == other_before
        assert (
            http.put(
                description_url,
                headers=headers,
                json={"description": "", "revision": 1},
            ).status_code
            == 200
        )
        assert (
            http.get(list_url, headers=headers).json()["items"][0][
                "dataset_description_preview"
            ]
            == ""
        )
        assert app.state.ingest.store.get("job", job["id"]) == job


def test_result_previews_support_existing_descriptions_and_warm_index(
    settings_factory, monkeypatch
):
    settings = settings_factory("server")
    app = create_app(settings, background=False)
    for number in range(3):
        job, _ = seed_result(app, f"result-{number}", created_at=number)
        # Descriptions written before list previews have no robot_type/job_id.
        app.state.ingest.store.put(
            "dataset_description",
            job["id"],
            {
                "id": job["id"],
                "description": f"介绍 {number}",
                "revision": 1,
                "updated_at": 1,
            },
        )
    lists = app.state.lists
    lists.index.warm()
    monkeypatch.setattr(
        app.state.ingest.store,
        "all",
        lambda *args, **kwargs: pytest.fail("warm list must not scan state"),
    )
    page = lists.page("results", "agilex_aloha", limit=1, offset=1)
    assert page["total"] == 3
    assert len(page["items"]) == 1
    assert page["items"][0]["id"] == "result-1"
    assert page["items"][0]["dataset_description_preview"] == "介绍 1"
    assert lists.index.version("dataset_description", "agilex_aloha") == 3
