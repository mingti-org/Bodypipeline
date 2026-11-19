import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from test_ingest import manifest
from test_processing import upload_sample

from pipeline.client.adapters.mock import MockAdapter
from pipeline.server.app import create_app
from pipeline.server.checks import QualityRunner
from pipeline.server.ingest import IngestService
from pipeline.server.jobs import JobRunner
from pipeline.shared.core import file_hash


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail("worker did not reach the expected state")


def seed(runner, identifier, created_at, **fields):
    row = {"id": identifier, "state": "QUEUED", "created_at": created_at, **fields}
    runner.store.put("job", identifier, row)
    return row


def raw_sample(runner, tmp_path):
    # Server release bundles need not include all Client Settings fields.
    raw = tmp_path / "input-raw"
    adapter = MockAdapter(runner.settings.profile())
    adapter.start("concurrency-sample", raw, "fold cloth")
    try:
        wait_for(lambda: adapter.count >= 3)
    finally:
        adapter.stop("concurrency-sample")
    sample, _ = manifest(runner.settings)
    sample["files"] = [
        {
            "path": path.relative_to(raw).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": file_hash(path),
        }
        for path in sorted(raw.rglob("*"))
        if path.is_file()
    ]
    return raw, sample


@pytest.fixture
def runner(settings_factory, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    settings = settings_factory("server", max_concurrent_jobs=2)
    worker = JobRunner(settings, IngestService(settings))
    yield worker
    worker.shutdown()


@pytest.mark.parametrize("limit", [2, 4])
def test_concurrent_jobs_respect_limit_queue_overflow_and_do_not_duplicate(
    runner, monkeypatch, limit
):
    runner.settings.max_concurrent_jobs = limit
    names = [f"job-{index}" for index in range(limit + 1)]
    releases = {name: threading.Event() for name in names}
    started = []
    active = set()
    peak = 0
    guard = threading.Lock()
    for index, name in enumerate(releases):
        seed(runner, name, index, runtime_id=f"runtime-{index}")

    def run(job):
        nonlocal peak
        with guard:
            started.append(job["id"])
            active.add(job["id"])
            peak = max(peak, len(active))
        try:
            assert releases[job["id"]].wait(5)
            runner.save(job, state="COMPLETED")
        finally:
            with guard:
                active.remove(job["id"])

    monkeypatch.setattr(runner, "run", run)
    try:
        with ThreadPoolExecutor(max_workers=4) as callers:
            results = list(callers.map(lambda _: runner.dispatch(), range(8)))
        assert sum(results) == 1
        wait_for(lambda: len(started) == limit)
        assert set(started) == set(names[:-1])
        assert runner.store.get("job", names[-1])["state"] == "QUEUED"
        releases[names[0]].set()
        wait_for(lambda: runner._active[names[0]].done())
        assert runner.dispatch()
        wait_for(lambda: names[-1] in started)
        assert not releases[names[1]].is_set()
        assert runner.store.get("job", names[1])["state"] == "RUNNING"
    finally:
        for release in releases.values():
            release.set()
        runner.shutdown()
    assert peak == limit
    assert sorted(started) == names
    assert all(row["state"] == "COMPLETED" for row in runner.store.all("job"))


def test_serial_limit_preserves_fifo(settings_factory, monkeypatch):
    settings = settings_factory("server", max_concurrent_jobs=1)
    worker = JobRunner(settings, IngestService(settings))
    release = threading.Event()
    started = []
    seed(worker, "oldest", 1, updated_at=100)
    seed(worker, "newest", 2)

    def run(job):
        started.append(job["id"])
        assert release.wait(5)
        worker.save(job, state="COMPLETED")

    monkeypatch.setattr(worker, "run", run)
    try:
        assert worker.dispatch()
        wait_for(lambda: started == ["oldest"])
        assert not worker.dispatch()
        assert not worker.tick()
        assert worker.store.get("job", "newest")["state"] == "QUEUED"
        release.set()
        wait_for(lambda: worker._active["oldest"].done())
        assert worker.dispatch()
        wait_for(lambda: started == ["oldest", "newest"])
    finally:
        release.set()
        worker.shutdown()


def test_worker_failure_frees_slot_without_interrupting_neighbor(runner, monkeypatch):
    release = threading.Event()
    started = []
    for index, name in enumerate(("bad", "neighbor", "next")):
        seed(runner, name, index)

    def run(job):
        started.append(job["id"])
        if job["id"] == "bad":
            raise RuntimeError("conversion failed")
        assert release.wait(5)
        runner.save(job, state="COMPLETED")

    monkeypatch.setattr(runner, "run", run)
    try:
        runner.dispatch()
        wait_for(lambda: runner._active["bad"].done())
        assert runner.store.get("job", "bad")["state"] == "FAILED"
        assert runner.dispatch()
        wait_for(lambda: "next" in started)
        assert runner.store.get("job", "neighbor")["state"] == "RUNNING"
    finally:
        release.set()


def test_overlapping_source_datasets_wait_while_unrelated_job_runs(
    runner, monkeypatch, tmp_path
):
    releases = {name: threading.Event() for name in ("first", "overlap", "unrelated")}
    started = []
    for name in ("source", "alias"):
        seed(
            runner, name, -1, state="COMPLETED", dataset_path=str(tmp_path / "dataset")
        )
    seed(runner, "first", 0, source_job_id="source")
    seed(runner, "overlap", 1, source_job_ids=["alias"])
    seed(runner, "unrelated", 2)

    def run(job):
        started.append(job["id"])
        assert releases[job["id"]].wait(5)
        runner.save(job, state="COMPLETED")

    monkeypatch.setattr(runner, "run", run)
    try:
        runner.dispatch()
        wait_for(lambda: len(started) == 2)
        assert set(started) == {"first", "unrelated"}
        assert runner.store.get("job", "overlap")["state"] == "QUEUED"
        releases["first"].set()
        wait_for(lambda: runner._active["first"].done())
        assert runner.dispatch()
        wait_for(lambda: "overlap" in started)
    finally:
        for release in releases.values():
            release.set()


def test_submit_failure_returns_claim_to_queue(runner):
    seed(runner, "first", 0)
    seed(runner, "second", 1)

    def submit(*args):
        raise RuntimeError("executor unavailable")

    runner._executor = SimpleNamespace(submit=submit, shutdown=lambda **kwargs: None)
    with pytest.raises(RuntimeError, match="executor unavailable"):
        runner.dispatch()
    assert not runner._active
    assert not runner._source_reservations
    assert all(row["state"] == "QUEUED" for row in runner.store.all("job"))


def test_dataset_readers_and_derived_jobs_reserve_the_same_source(runner, monkeypatch):
    seed(runner, "source", -1, state="COMPLETED")
    reader = QualityRunner(runner.settings, runner.store, runner.command)
    runner.store.put(
        "check", "check", {"id": "check", "job_id": "source", "state": "QUEUED"}
    )
    release = threading.Event()
    entered = threading.Event()

    def run_check(row):
        entered.set()
        assert release.wait(5)
        reader.save(row, state="COMPLETED")

    monkeypatch.setattr(reader, "run", run_check)
    monkeypatch.setattr(runner, "run", lambda job: runner.save(job, state="COMPLETED"))
    with ThreadPoolExecutor(max_workers=1) as caller:
        future = caller.submit(runner.dataset_tick, reader, "check")
        try:
            assert entered.wait(5)
            seed(runner, "derived", 1, source_job_id="source")
            seed(runner, "unrelated", 2)
            assert runner.dispatch()
            wait_for(lambda: runner._active["unrelated"].done())
            assert runner.store.get("job", "derived")["state"] == "QUEUED"
        finally:
            release.set()
        assert future.result(timeout=5)
    assert runner.dispatch()
    wait_for(lambda: runner._active["derived"].done())


def test_dataset_reader_waits_for_active_derived_job(runner, monkeypatch):
    seed(runner, "source", -1, state="COMPLETED")
    seed(runner, "derived", 0, source_job_id="source")
    reader = QualityRunner(runner.settings, runner.store, runner.command)
    runner.store.put(
        "check", "check", {"id": "check", "job_id": "source", "state": "QUEUED"}
    )
    release = threading.Event()

    def run(job):
        assert release.wait(5)
        runner.save(job, state="COMPLETED")

    monkeypatch.setattr(runner, "run", run)
    monkeypatch.setattr(reader, "run", lambda row: reader.save(row, state="COMPLETED"))
    try:
        assert runner.dispatch()
        assert not runner.dataset_tick(reader, "check")
        assert runner.store.get("check", "check")["state"] == "QUEUED"
    finally:
        release.set()
    wait_for(lambda: runner._active["derived"].done())
    runner.dispatch()
    assert runner.dataset_tick(reader, "check")


def test_dataset_tick_supports_preview_worker_callbacks(runner):
    seed(runner, "source", 0, state="COMPLETED")
    runner.store.put(
        "trim_preview",
        "preview",
        {"id": "preview", "job_id": "source", "state": "QUEUED"},
    )
    stages = []

    def save(row, **updates):
        row.update(updates)
        runner.store.put("trim_preview", row["id"], row)

    def run(row):
        assert runner._source_reservations
        stages.append(row["state"])
        save(row, state="COMPLETED")

    assert runner.dataset_tick(object(), "trim_preview", save=save, run=run)
    assert stages == ["RUNNING"]
    assert not runner._source_reservations


def test_shutdown_interrupts_all_active_jobs_releases_only_owned_allocations(
    runner, monkeypatch
):
    started = []
    cancelled = []
    for index, name in enumerate(("first", "second", "queued")):
        seed(runner, name, index)

    def run(job):
        runner.allocation_id = "allocation-" + job["id"]
        started.append(job["id"])
        assert runner.shutdown_requested.wait(5)
        raise InterruptedError("server stopping")

    monkeypatch.setattr(runner, "run", run)
    monkeypatch.setattr(
        runner, "_cancel_allocation", lambda *args: cancelled.append(args)
    )
    runner.dispatch()
    wait_for(lambda: len(started) == 2)
    assert runner.allocation_id is None
    runner.shutdown()
    assert not runner._active
    assert sorted(cancelled) == [("allocation-first",), ("allocation-second",)]
    assert runner.store.get("job", "queued")["state"] == "QUEUED"
    assert all(
        runner.store.get("job", name)["state"] == "INTERRUPTED" for name in started
    )
    assert not runner.dispatch()
    assert not runner.tick()


def test_demo_conversion_pipeline_keeps_outputs_independent(runner, tmp_path):
    raw, sample = raw_sample(runner, tmp_path)
    upload_id = upload_sample(runner.ingest, raw, sample)
    jobs = [
        runner.enqueue(upload_id, instruction=text)
        for text in ("first task", "second task")
    ]
    assert runner.dispatch()
    wait_for(lambda: all(future.done() for future in runner._active.values()))
    results = [runner.store.get("job", job["id"]) for job in jobs]
    assert [row["state"] for row in results] == ["DEMO_READY", "DEMO_READY"]
    assert results[0]["work_path"] != results[1]["work_path"]
    assert (
        results[0]["alignment_report"]["frames"]
        == results[1]["alignment_report"]["frames"]
    )
    assert all(
        len(list(Path(row["work_path"], "aligned").iterdir())) == 1 for row in results
    )


def test_real_letools_jobs_convert_in_parallel(
    runner, tmp_path, letools_binary, monkeypatch
):
    runner.settings.processing_mode = "real"
    runner.settings.letools_bin = letools_binary
    raw, sample = raw_sample(runner, tmp_path)
    upload_id = upload_sample(runner.ingest, raw, sample)
    jobs = [
        runner.enqueue(upload_id, instruction=text, run_doctor=False)
        for text in ("first task", "second task")
    ]
    both_converting = threading.Barrier(2)
    command = runner.command

    def parallel_command(args, work, name, **kwargs):
        if name == "convert":
            both_converting.wait(timeout=30)
        return command(args, work, name, **kwargs)

    monkeypatch.setattr(runner, "command", parallel_command)
    assert runner.dispatch()
    wait_for(
        lambda: all(future.done() for future in runner._active.values()), timeout=120
    )
    results = [runner.store.get("job", job["id"]) for job in jobs]
    assert [row["state"] for row in results] == ["CONVERTED", "CONVERTED"], results
    assert results[0]["dataset_path"] != results[1]["dataset_path"]
    for result in results:
        dataset = Path(result["dataset_path"])
        info = json.loads((dataset / "meta/info.json").read_text())
        assert info["total_episodes"] == 1
        assert info["total_frames"] == result["alignment_report"]["frames"]
        assert info["codebase_version"] == "v3.0"
        assert json.loads(Path(result["work_path"], "validate.json").read_text())[
            "valid"
        ]


def test_restart_recovers_all_running_jobs_without_restarting_them(runner):
    for index, name in enumerate(("first", "second")):
        seed(runner, name, index, state="RUNNING", work_path=f"/work/{name}")
    seed(runner, "queued", 2)
    runner.recover()
    assert runner.store.get("job", "queued")["state"] == "QUEUED"
    for name in ("first", "second"):
        row = runner.store.get("job", name)
        assert row["state"] == "INTERRUPTED"
        assert row["work_path"] == f"/work/{name}"


def test_slurm_allocations_are_unique_and_released_independently(
    runner, monkeypatch, tmp_path
):
    runner.settings.slurm = {
        "enabled": True,
        "job_name": "conversion",
        "cpus_per_task": 32,
    }
    allocations = {}
    cancelled = []
    guard = threading.Lock()
    barrier = threading.Barrier(2)

    def popen(argv, **kwargs):
        assert argv[argv.index("--chdir") + 1] == str(kwargs["cwd"].resolve())
        name = argv[argv.index("--job-name") + 1]
        with guard:
            allocations[name] = (str(100 + len(allocations)), kwargs["cwd"])
        return SimpleNamespace(poll=lambda: 0, returncode=0)

    def run(argv, **kwargs):
        if argv[0].endswith("squeue"):
            name = argv[argv.index("--name") + 1]
            identifier, work = allocations[name]
            return SimpleNamespace(
                stdout=f"{identifier}|RUNNING|{work}\n", returncode=0
            )
        assert argv[0].endswith("scancel")
        cancelled.append(argv[1:])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "pipeline.server.jobs.executable_path", lambda name: f"/usr/bin/{name}"
    )
    monkeypatch.setattr("pipeline.server.jobs.subprocess.Popen", popen)
    monkeypatch.setattr("pipeline.server.jobs.subprocess.run", run)

    def allocate(index):
        work = tmp_path / str(index)
        work.mkdir()
        identifier = runner._open_allocation(work)
        barrier.wait(timeout=5)
        assert runner.allocation_id == identifier
        runner._close_allocation()
        return identifier

    with ThreadPoolExecutor(max_workers=2) as callers:
        identifiers = list(callers.map(allocate, range(2)))
    assert len(allocations) == len(set(identifiers)) == 2
    assert all(name.startswith("conversion-") for name in allocations)
    assert sorted(cancelled) == [["100"], ["101"]]
    assert runner.allocation_id is None


def test_inherited_allocation_is_not_cancelled(runner, monkeypatch, tmp_path):
    runner.settings.slurm = {
        "enabled": True,
        "cpus_per_task": 32,
        "conversion_workers": 8,
    }
    monkeypatch.setenv("SLURM_JOB_ID", "parent")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "32")
    cancelled = []
    monkeypatch.setattr(
        runner, "_cancel_allocation", lambda *args: cancelled.append(args)
    )
    monkeypatch.setattr(
        "pipeline.server.jobs.executable_path", lambda name: f"/usr/bin/{name}"
    )
    assert runner._open_allocation(tmp_path) == "parent"
    runner._close_allocation()
    assert cancelled == []
    assert runner._conversion_parallelism()["video_codec_threads"] == 2
    args = runner._scheduled(["tool"], allocation="parent")
    assert args[args.index("--cpus-per-task") + 1] == "16"
    assert "--exact" in args


def test_slurm_budget_uses_requested_cpus_instead_of_server_affinity(
    runner, monkeypatch
):
    monkeypatch.setattr(
        os, "sched_getaffinity", lambda _: set(range(192)), raising=False
    )
    runner.settings.slurm = {
        "enabled": True,
        "cpus_per_task": 32,
        "conversion_workers": 8,
    }
    assert runner._conversion_parallelism() == {
        "convert_workers": 8,
        "video_workers": 8,
        "video_codec_threads": 4,
    }
    runner.settings.slurm["cpus_per_task"] = 2
    assert runner._conversion_parallelism()["convert_workers"] == 2
    assert runner._conversion_parallelism()["video_codec_threads"] == 1
    runner.settings.slurm = {"enabled": False, "conversion_workers": 8}
    assert runner._conversion_parallelism()["video_codec_threads"] == 12


def test_slurm_shutdown_cancels_only_the_pending_request(runner, monkeypatch, tmp_path):
    runner.settings.slurm = {"enabled": True}
    process_started = threading.Event()
    cancelled = []
    real_popen = subprocess.Popen

    def popen(argv, **kwargs):
        process = real_popen(
            [sys.executable, "-c", "import time; time.sleep(60)"], **kwargs
        )
        process_started.set()
        return process

    monkeypatch.setattr(
        "pipeline.server.jobs.executable_path", lambda name: f"/usr/bin/{name}"
    )
    monkeypatch.setattr("pipeline.server.jobs.subprocess.Popen", popen)
    monkeypatch.setattr(
        runner, "_cancel_allocation", lambda *args: cancelled.append(args)
    )
    with ThreadPoolExecutor(max_workers=1) as callers:
        future = callers.submit(runner._open_allocation, tmp_path)
        assert process_started.wait(5)
        runner.shutdown_requested.set()
        with pytest.raises(InterruptedError):
            future.result(timeout=5)
    assert len(cancelled) == 1
    assert cancelled[0][0] == "--name"
    assert cancelled[0][1].startswith("bodypipeline-convert-")
    assert runner.allocation_id is None


def test_quality_failure_releases_allocation(runner, monkeypatch, tmp_path):
    opened, closed = [], []
    checks = QualityRunner(
        runner.settings,
        runner.store,
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("tool failed")),
        open_allocation=lambda work: opened.append(work) or "check-allocation",
        close_allocation=lambda: closed.append(True),
    )
    seed(runner, "result", 0, state="COMPLETED")
    monkeypatch.setattr(
        "pipeline.server.checks.dataset_for_job", lambda *args: tmp_path
    )
    monkeypatch.setattr(
        "pipeline.server.checks.executable_path", lambda _: "/opt/letools"
    )
    checks.run({"id": "check", "job_id": "result", "state": "QUEUED"})
    assert opened and closed == [True]
    assert runner.store.get("check", "check")["state"] == "FAILED"


@pytest.mark.parametrize("value", [True, 0, -1, 17, 1.5, "2", None])
def test_parallelism_configuration_rejects_invalid_values(settings_factory, value):
    with pytest.raises(ValueError, match="max_concurrent_jobs"):
        settings_factory("server", max_concurrent_jobs=value)


def test_background_dispatch_continues_while_other_worker_is_busy(
    settings_factory, monkeypatch
):
    settings = settings_factory("server")
    app = create_app(settings)
    worker = app.state.runner
    source_started = threading.Event()
    started = []

    seed(worker, "source", -1, state="COMPLETED")
    worker.store.put(
        "check", "check", {"id": "check", "job_id": "source", "state": "QUEUED"}
    )

    def blocking_check(row):
        source_started.set()
        assert worker.shutdown_requested.wait(5)
        app.state.checks.save(row, state="INTERRUPTED")

    def run(job):
        started.append(job["id"])
        assert worker.shutdown_requested.wait(5)
        raise InterruptedError("server stopping")

    monkeypatch.setattr(app.state.checks, "run", blocking_check)
    monkeypatch.setattr(app.state.viewer, "autostart", lambda: None)
    monkeypatch.setattr(worker, "run", run)
    with TestClient(app) as http:
        assert source_started.wait(5)
        for index, name in enumerate(("first", "second", "third", "fourth", "queued")):
            seed(worker, name, index)
        wait_for(lambda: len(started) == 4)
        assert http.get("/healthz").status_code == 200
        config = http.get(
            "/api/config", headers={"Authorization": "Bearer " + settings.token}
        )
        assert config.status_code == 200
        assert config.json()["max_concurrent_jobs"] == 4
    assert worker.store.get("job", "queued")["state"] == "QUEUED"
    assert all(
        worker.store.get("job", name)["state"] == "INTERRUPTED" for name in started
    )
    assert not worker._active
