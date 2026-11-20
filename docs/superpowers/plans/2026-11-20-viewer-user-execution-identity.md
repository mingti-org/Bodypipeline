# Viewer User Execution Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Visualizer compute tasks execute through the submitting user's existing H800 ControlMaster while preserving service-account behavior for machine and legacy jobs.

**Architecture:** Use `request.state.principal` as the only identity input, persist the standard execution metadata on each new job, and bind that metadata through `JobRunner.bind_record_session()` before any remote worker starts. Manual trim, instruction apply, review export, and episode scoring share the same fail-closed command and permission boundary.

**Tech Stack:** Python 3.12, FastAPI, pytest, OpenSSH ControlMaster, sshpass, Slurm/srun, LeTools workers, JSON-backed Store

**Spec:** `docs/superpowers/specs/2026-11-20-embedded-viewer-user-execution-design.md`

## Global Constraints

- Authenticated identity comes only from `request.state.principal`.
- Usernames, UIDs, SSH execution handles, and login nodes are never accepted from request JSON, query parameters, or client-controlled headers.
- A personal execution job stores `owner_user_id`, `owner_uid`, `submitted_as`, `slurm_user`, `slurm_execution_id`, and `submission_auth` from `Principal.job_metadata()`.
- `slurm_execution_id` is an internal execution handle and never appears in API responses, browser state, UI code, logs, error messages, receipts, or checkout artifacts.
- A missing or expired personal SSH session fails closed and never falls back to the Server account or another user.
- Service-token, device-token, and internal jobs retain existing `service-account` behavior; legacy jobs without `slurm_execution_id` remain resumable.
- Deduplication keys for personal jobs include the submitting username.
- Remote workload execution uses the submitter's OpenSSH ControlMaster; queue state, indexes, locks, and publication bookkeeping remain Server-owned.
- File permission probes use argv-based subprocess calls and never interpolate user paths into a shell string.
- Cookie-authenticated mutating requests require a same-origin `Origin` header.
- Passwords, SSH control sockets, and `slurm_execution_id` are never logged or persisted in UI state.
- The standalone 8082 compatibility path may retain service-token behavior during the rollout window; the per-user execution guarantee applies to the embedded 8081 surface.

---

### Task 1: Redact private execution handles at the API boundary

**Files:**
- Modify: `pipeline/shared/identity.py:1-80`
- Modify: `pipeline/server/app.py:52-90`
- Modify: `pipeline/server/app.py:650-685`
- Modify: `pipeline/server/app.py:760-940`
- Modify: `pipeline/server/app.py:990-1035`
- Modify: `pipeline/server/checks.py:55-85`
- Modify: `pipeline/server/analyses.py:225-265`
- Modify: `pipeline/server/annotations.py:515-550`
- Modify: `pipeline/server/annotations.py:710-745`
- Test: `tests/test_user_identity.py`

**Interfaces:**
- Produces: `public_execution_record(row: dict) -> dict`
- Produces: every API-returned task, check, analysis, trim preview, and apply record removes `slurm_execution_id`
- Consumes: existing persisted identity metadata

- [ ] **Step 1: Write the failing API redaction test**

```python
def test_task_apis_never_return_private_execution_handle(settings_factory):
    settings = settings_factory("server")
    app = create_app(settings, background=False)
    app.state.runner.store.put(
        "job",
        "job-a",
        {
            "id": "job-a",
            "state": "QUEUED",
            "created_at": 1,
            "owner_user_id": "alice",
            "submitted_as": "alice",
            "slurm_user": "alice",
            "slurm_execution_id": "private-execution-token",
            "submission_auth": "h800-ssh",
        },
    )
    app.state.runner.store.put(
        "trim_preview",
        "preview-a",
        {
            "id": "preview-a",
            "job_id": "job-a",
            "state": "QUEUED",
            "created_at": 1,
            "slurm_execution_id": "private-execution-token",
        },
    )
    headers = {"Authorization": "Bearer " + settings.token}

    with TestClient(app, headers=headers) as client:
        job = client.get("/api/jobs/job-a")
        previews = client.get("/api/jobs/job-a/trim-previews")

    assert job.status_code == 200
    assert "slurm_execution_id" not in job.json()
    assert previews.status_code == 200
    assert "private-execution-token" not in previews.text
```

- [ ] **Step 2: Run the test and confirm it leaks the handle**

```bash
pytest tests/test_user_identity.py::test_task_apis_never_return_private_execution_handle -v
```

Expected: FAIL because `job_response()` returns stored job fields unchanged and
trim preview APIs return stored rows directly.

- [ ] **Step 3: Add one shared redaction helper**

```python
# pipeline/shared/identity.py
PRIVATE_EXECUTION_FIELDS = frozenset({"slurm_execution_id"})


def public_execution_record(row: dict) -> dict:
    return {
        key: value
        for key, value in row.items()
        if key not in PRIVATE_EXECUTION_FIELDS
    }
```

Apply it to every outward-facing record produced from `job`, `check`,
`analysis`, `annotation`, `trim_preview`, and `instruction_apply` storage. At
minimum, update `job_response()`, the check/analysis listing handlers, trim
preview endpoints, annotation API responses, and instruction apply `_public()`.
When child rows are created by copying execution metadata from a parent, copy
only ownership and authentication fields and omit `slurm_execution_id`
entirely. Internal execution code continues reading the original store row.

Do not recursively remove similarly named strings from reports or logs. The
private field is a top-level execution record field. Any report that copies
this field must stop copying it.

- [ ] **Step 4: Run identity and API tests**

```bash
pytest tests/test_user_identity.py tests/test_server_api.py tests/test_workbench_lists.py -q
```

Expected: PASS. Public responses retain `owner_user_id`, `submitted_as`,
`slurm_user`, and `submission_auth`, while `slurm_execution_id` is absent.

- [ ] **Step 5: Commit**

```bash
git add \
  pipeline/shared/identity.py \
  pipeline/server/app.py \
  pipeline/server/checks.py \
  pipeline/server/analyses.py \
  pipeline/server/annotations.py \
  tests/test_user_identity.py
git commit -m "fix(server): redact private execution handles"
```

### Task 2: Add a fail-closed captured-command and permission bridge

**Files:**
- Modify: `pipeline/shared/identity.py:27-57`
- Modify: `pipeline/server/jobs.py:60-125`
- Modify: `pipeline/server/jobs.py:410-455`
- Modify: `tests/test_user_identity.py`

**Interfaces:**
- Produces: `JobRunner.bind_principal(principal) -> contextmanager[UserSession | None]`
- Produces: `JobRunner.run_captured(argv, *, timeout, remote_env=None) -> subprocess.CompletedProcess`
- Produces: `JobRunner.require_principal_path_access(principal, dataset, *, writable=False) -> None`
- Consumes: `Principal.job_metadata()`, `SlurmIdentityBroker.session_for_execution()`, and `SlurmIdentityBroker.run()`

- [ ] **Step 1: Write the failing remote-capture and permission tests**

```python
def test_captured_command_uses_bound_user_without_local_fallback(
    settings_factory, tmp_path
):
    settings = settings_factory("server")
    store = Store(settings.root / "state")

    class Broker:
        def __init__(self):
            self.calls = []

        def session_for_execution(self, execution_id):
            assert execution_id == "execution-a"
            return SimpleNamespace(username="alice", uid=1001)

        def run(self, session, argv, **kwargs):
            self.calls.append((session, argv, kwargs))
            return SimpleNamespace(returncode=0, stdout='{"ok": true}', stderr="")

    broker = Broker()
    runner = JobRunner(
        settings,
        SimpleNamespace(store=store),
        user_sessions=broker,
    )
    metadata = {
        "owner_user_id": "alice",
        "owner_uid": 1001,
        "submitted_as": "alice",
        "slurm_user": "alice",
        "slurm_execution_id": "execution-a",
        "submission_auth": "h800-ssh",
    }

    with runner.bind_record_session(metadata):
        result = runner.run_captured(
            ["/workspace/.venv/bin/python", "worker.py"],
            timeout=30,
        )

    assert result.stdout == '{"ok": true}'
    assert broker.calls[0][1] == [
        "/workspace/.venv/bin/python",
        "worker.py",
    ]
```

```python
def test_permission_probe_uses_remote_test_argv(
    settings_factory, tmp_path, monkeypatch
):
    settings = settings_factory(
        "server",
        user_sessions={"enabled": True, "login_nodes": ["H800-node11"]},
    )
    store = Store(settings.root / "state")
    broker = SlurmIdentityBroker(settings)
    session = UserSession(
        session_id="opaque",
        execution_id="execution-a",
        username="alice",
        uid=1001,
        login_node="H800-node11",
        control_path=tmp_path / "control",
        expires_at=now() + 60,
    )
    with broker._lock:
        broker._by_token[session.session_id] = session
        broker._by_execution[session.execution_id] = session
    runner = JobRunner(
        settings,
        SimpleNamespace(store=store),
        user_sessions=broker,
    )
    calls = []

    def run(active, argv, **kwargs):
        calls.append((active, argv))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(broker, "run", run)
    writable = settings.root / "dataset"
    (writable / "meta").mkdir(parents=True)
    (writable / "meta/info.json").write_text("{}")
    principal = session.principal()

    runner.require_principal_path_access(
        principal, writable, writable=True
    )

    assert all(active is session for active, _ in calls)
    assert calls[-1][1][:2] == ["/usr/bin/test", "-w"]
    assert str(writable) == calls[-1][1][-1]
```

Add the fail-closed ownership tests:

```python
def test_personal_metadata_requires_execution_handle():
    principal = Principal(
        username="alice",
        uid=1001,
        auth_method="h800-ssh",
        execution_id=None,
    )
    with pytest.raises(DomainError, match="H800 登录会话"):
        JobRunner.submission_metadata(principal)


def test_device_identity_remains_service_account():
    device = Principal(
        username="device:tianji-01",
        auth_method="device-token",
        is_admin=False,
    )
    assert JobRunner.submission_metadata(device) == {
        "owner_user_id": None,
        "owner_uid": None,
        "submitted_as": "service-account",
        "slurm_user": None,
        "slurm_execution_id": None,
        "submission_auth": "device-token",
    }


def test_execution_handle_cannot_cross_users(settings_factory):
    settings = settings_factory("server")
    store = Store(settings.root / "state")

    class Broker:
        def session_for_execution(self, execution_id):
            return SimpleNamespace(username="bob", uid=1002)

    runner = JobRunner(
        settings,
        SimpleNamespace(store=store),
        user_sessions=Broker(),
    )
    row = {
        "owner_user_id": "alice",
        "owner_uid": 1001,
        "submitted_as": "alice",
        "slurm_user": "alice",
        "slurm_execution_id": "execution-a",
        "submission_auth": "h800-ssh",
    }

    with pytest.raises(DomainError, match="提交人不匹配"):
        with runner.bind_record_session(row):
            pass


def test_captured_command_requires_explicit_binding(settings_factory):
    settings = settings_factory("server")
    store = Store(settings.root / "state")
    runner = JobRunner(
        settings,
        SimpleNamespace(store=store),
    )
    with pytest.raises(DomainError, match="必须绑定提交人身份"):
        runner.run_captured(["true"], timeout=1)
```

- [ ] **Step 2: Run the tests and confirm the interface is absent**

```bash
pytest \
  tests/test_user_identity.py::test_captured_command_uses_bound_user_without_local_fallback \
  tests/test_user_identity.py::test_permission_probe_uses_remote_test_argv \
  tests/test_user_identity.py::test_personal_metadata_requires_execution_handle \
  tests/test_user_identity.py::test_device_identity_remains_service_account \
  tests/test_user_identity.py::test_execution_handle_cannot_cross_users \
  tests/test_user_identity.py::test_captured_command_requires_explicit_binding -v
```

Expected: FAIL because the interfaces do not exist and the current metadata
builder does not enforce the personal/device/ownership boundary.

- [ ] **Step 3: Implement fail-closed metadata, session binding, and command execution**

Change `Principal.job_metadata()` so only a `h800-ssh` identity can produce a
personal execution record. Admin, browser-token, service-token, and device-token
principals retain the Server process account:

```python
def job_metadata(self) -> dict:
    if self.is_admin or self.auth_method != "h800-ssh":
        return {
            "owner_user_id": None,
            "owner_uid": None,
            "submitted_as": "service-account",
            "slurm_user": None,
            "slurm_execution_id": None,
            "submission_auth": self.auth_method,
        }
    if not self.execution_id:
        raise DomainError(
            "任务发起者的 H800 登录会话不可用；请重新登录",
            401,
        )
    return {
        "owner_user_id": self.username,
        "owner_uid": self.uid,
        "submitted_as": self.username,
        "slurm_user": self.username,
        "slurm_execution_id": self.execution_id,
        "submission_auth": self.auth_method,
    }
```

Validate the broker session against the persisted owner before exposing it to
any worker:

```python
@contextmanager
def bind_record_session(self, row):
    execution_id = row.get("slurm_execution_id")
    personal = row.get("submission_auth") == "h800-ssh"
    session = None
    if execution_id:
        if self.user_sessions is None:
            raise DomainError(
                "任务需要 H800 用户会话，但服务未配置用户登录", 503
            )
        session = self.user_sessions.session_for_execution(execution_id)
        if session is None:
            raise DomainError(
                "任务发起者的 H800 登录会话已过期；请重新登录并重新提交任务",
                401,
            )
        expected = {
            "username": row.get("slurm_user"),
            "uid": row.get("owner_uid"),
        }
        actual = {
            "username": session.username,
            "uid": session.uid,
        }
        if (
            row.get("owner_user_id") != session.username
            or row.get("submitted_as") != session.username
            or expected != actual
        ):
            raise DomainError(
                "任务执行会话与提交人不匹配；已拒绝执行", 403
            )
    elif personal:
        raise DomainError(
            "任务发起者的 H800 登录会话不可用；请重新登录",
            401,
        )
    previous = self.execution_session
    previous_bound = getattr(self._execution_context, "bound", False)
    self._execution_context.session = session
    self._execution_context.bound = True
    try:
        yield session
    finally:
        self._execution_context.session = previous
        self._execution_context.bound = previous_bound
```

```python
@contextmanager
def bind_principal(self, principal):
    metadata = self.submission_metadata(principal)
    with self.bind_record_session(metadata) as session:
        yield session


def run_captured(self, argv, *, timeout, remote_env=None):
    command = [str(value) for value in argv]
    environment = self.settings.tool_environment()
    if not getattr(self._execution_context, "bound", False):
        raise DomainError(
            "任务执行前必须绑定提交人身份；已拒绝未绑定执行", 500
        )
    session = self.execution_session
    if session is None:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
        )
    return self.user_sessions.run(
        session,
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=environment,
        remote_env=remote_env,
    )
```

Implement path checks with `/usr/bin/test` calls:

```python
def require_principal_path_access(
    self, principal, dataset, *, writable=False
):
    if (
        principal is None
        or principal.is_admin
        or principal.auth_method != "h800-ssh"
    ):
        return
    dataset = Path(dataset).resolve()
    checks = [
        ["-x", str(dataset)],
        ["-r", str(dataset)],
        ["-r", str(dataset / "meta/info.json")],
    ]
    if writable:
        checks.append(["-w", str(dataset)])
        for relative in ("meta", "data", "videos"):
            path = dataset / relative
            if path.exists():
                checks.append(["-w", str(path)])
    with self.bind_principal(principal) as session:
        if session is None:
            raise DomainError("任务需要 H800 用户会话", 503)
        for check in checks:
            result = self.user_sessions.run(
                session,
                ["/usr/bin/test", *check],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode:
                raise DomainError(
                    f"H800 用户 {principal.username} 无权访问数据集路径",
                    403,
                )
```

If the session expires between the outer request and this probe,
`bind_record_session()` raises the existing 401 error. Do not catch it and do
not run the local fallback.

Update the existing
`test_slurm_command_uses_the_submitting_users_control_master` fixture so its
`bind_record_session()` row contains the full `alice` identity metadata. A row
that carries an execution handle without matching ownership fields must now be
rejected.

- [ ] **Step 4: Run the focused identity tests**

```bash
pytest tests/test_user_identity.py -q
```

Expected: PASS, including the existing password, ControlMaster, Slurm, and
submission metadata tests.

- [ ] **Step 5: Commit**

```bash
git add \
  pipeline/shared/identity.py \
  pipeline/server/jobs.py \
  tests/test_user_identity.py
git commit -m "feat(server): add user execution command bridge"
```

### Task 3: Attribute manual trim to the submitting user

**Files:**
- Modify: `pipeline/server/app.py:770-790`
- Modify: `pipeline/server/trims.py:848-960`
- Modify: `pipeline/server/trims.py:1070-1090`
- Test: `tests/test_stationary_trim.py`
- Test: `tests/test_instruction_review.py`

**Interfaces:**
- Consumes: `trim_start(jid: str, body: ManualTrimRequest, request: Request)`
- Produces: `TrimRunner.enqueue_manual(..., principal: Principal | None = None)`
- Produces: personal `operation="manual_trim"` rows with standard execution metadata
- Consumes: `JobRunner.require_principal_path_access()`

- [ ] **Step 1: Write the failing manual-trim identity test**

```python
def test_manual_trim_persists_submitting_user_and_redacts_handle(
    settings_factory, monkeypatch
):
    settings = settings_factory("server", letools_bin=sys.executable)
    app = create_app(settings, background=False)
    seed(app)
    captured = []

    monkeypatch.setattr(
        app.state.runner,
        "require_principal_path_access",
        lambda principal, path, writable: captured.append(
            (principal.username, path, writable)
        ),
    )
    principal = Principal(
        username="alice",
        uid=1001,
        login_node="H800-node11",
        execution_id="execution-a",
        auth_method="h800-ssh",
    )
    job = app.state.trims.enqueue_manual(
        "source",
        0,
        [[1, 2]],
        principal=principal,
    )
    stored = app.state.runner.store.get("job", job["id"])

    assert captured[0][0] == "alice"
    assert stored["owner_user_id"] == "alice"
    assert stored["submitted_as"] == "alice"
    assert stored["slurm_user"] == "alice"
    assert stored["slurm_execution_id"] == "execution-a"


def test_manual_trim_dedup_is_per_submitter(settings_factory, monkeypatch):
    settings = settings_factory("server", letools_bin=sys.executable)
    app = create_app(settings, background=False)
    seed(app)
    monkeypatch.setattr(
        app.state.runner,
        "require_principal_path_access",
        lambda principal, path, writable: None,
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

    first = app.state.trims.enqueue_manual(
        "source", 0, [[1, 2]], principal=alice
    )
    same = app.state.trims.enqueue_manual(
        "source", 0, [[1, 2]], principal=alice
    )
    other = app.state.trims.enqueue_manual(
        "source", 0, [[1, 2]], principal=bob
    )

    assert same["id"] == first["id"]
    assert other["id"] != first["id"]
```

- [ ] **Step 2: Run the tests and confirm identity is missing**

```bash
pytest tests/test_stationary_trim.py::test_manual_trim_persists_submitting_user_and_redacts_handle tests/test_stationary_trim.py::test_manual_trim_dedup_is_per_submitter -v
```

Expected: FAIL because `enqueue_manual()` has no `principal` parameter and its
deduplication key has no submitter.

- [ ] **Step 3: Thread the authenticated principal from HTTP to queue**

Change the Server endpoint to:

```python
@app.post("/api/jobs/{jid}/manual-trims", status_code=202)
def manual_trim_start(
    jid: str, body: ManualTrimRequest, request: Request
):
    accessible_job(jid, request, action="裁剪")
    return job_response(
        trims.enqueue_manual(
            jid,
            body.episode_index,
            body.remove_ranges,
            name=body.name,
            run_doctor=body.run_doctor,
            in_place=body.in_place,
            principal=principal(request),
        )
    )
```

Also update the native embedded Viewer route created by the gateway plan so
the browser surface does not bypass identity assignment:

```python
@app.post("/api/manual-trims/{dataset_id}", status_code=202)
def embedded_manual_trim_start(
    dataset_id: str,
    body: ManualTrimRequest,
    request: Request,
):
    preview = registered_preview(viewer_transport, dataset_id)
    return job_response(
        trims.enqueue_manual(
            preview["job_id"],
            body.episode_index,
            body.remove_ranges,
            name=body.name,
            run_doctor=body.run_doctor,
            in_place=body.in_place,
            principal=principal(request),
        )
    )
```

Change `TrimRunner.enqueue_manual()`:

```python
def enqueue_manual(
    self,
    source_job_id: str,
    episode_index: int,
    remove_ranges,
    name: str | None = None,
    run_doctor: bool = False,
    in_place: bool = True,
    *,
    principal: Principal | None = None,
) -> dict:
    submitter = self.runner.submission_metadata(principal)["submitted_as"]
    # Existing validation and dataset locks run before these additions.
    self.runner.require_principal_path_access(
        principal,
        source_path,
        writable=in_place,
    )
    key = "manual-trim:" + digest(
        {
            "source_job_id": source_job_id,
            "episode_index": episode_index,
            "remove_ranges": ranges,
            "source_snapshot": snapshot,
            "name": output_name,
            "run_doctor": run_doctor,
            "in_place": in_place,
            "letools": self.settings.letools_bin,
            "submitted_by": submitter,
        }
    )
```

Add the standard metadata to the new job:

```python
job = {
    "id": str(uuid.uuid4()),
    "key": key,
    "operation": "manual_trim",
    "source_kind": "lerobot_manual_trim",
    # Existing source, dataset, format, and trim fields stay unchanged.
    **self.runner.submission_metadata(principal),
}
```

The existing `JobRunner._execute_job()` already wraps every queued job in
`bind_record_session(job)`. Verify this call remains outside `trims.run(job)`;
the remote LeTools worker then executes over the user's ControlMaster.

- [ ] **Step 4: Run manual trim, instruction conflict, and identity tests**

```bash
pytest tests/test_stationary_trim.py tests/test_instruction_review.py tests/test_user_identity.py -q
```

Expected: PASS. Existing service-token tests still create
`submitted_as="service-account"` rows. Two users get separate active jobs.

- [ ] **Step 5: Commit**

```bash
git add pipeline/server/app.py pipeline/server/trims.py tests/test_stationary_trim.py tests/test_instruction_review.py
git commit -m "feat(server): execute manual trims as submitter"
```

### Task 4: Execute instruction apply through the submitter's SSH session

**Files:**
- Modify: `pipeline/server/app.py:217-235`
- Modify: `pipeline/server/app.py:900-930`
- Modify: `pipeline/server/instruction_worker.py:24-420`
- Test: `tests/test_instruction_api.py`
- Test: `tests/test_instruction_review.py`
- Test: `tests/test_user_identity.py`

**Interfaces:**
- Produces: `InstructionApplyRunner(settings, store, review, runner: JobRunner)`
- Produces: `InstructionApplyRunner.enqueue(..., principal: Principal | None = None)`
- Consumes: `JobRunner.bind_record_session()`
- Consumes: `JobRunner.run_captured()`
- Replaces: direct `subprocess.run()` in `InstructionApplyRunner._worker()`

- [ ] **Step 1: Write the failing worker-binding tests**

```python
def test_instruction_apply_records_submitter(
    instruction_app, monkeypatch
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
    monkeypatch.setattr(
        runner,
        "require_principal_path_access",
        lambda principal, path, writable: None,
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

    assert row["submitted_as"] == "alice"
    assert row["slurm_user"] == "alice"
    assert row["slurm_execution_id"] == "execution-a"
    assert "slurm_execution_id" not in queued
```

```python
def test_instruction_worker_uses_bound_remote_command(
    settings_factory, monkeypatch
):
    settings = settings_factory("server")
    store = Store(settings.root / "state")
    review = SimpleNamespace()

    class FakeRunner:
        execution_session = object()

        def run_captured(self, argv, *, timeout, remote_env=None):
            assert argv[1].endswith("instruction_writer_worker.py")
            assert argv[2] == "recover"
            assert timeout == settings.command_timeout
            return SimpleNamespace(
                returncode=0,
                stdout='{"operation_id": "apply-a"}',
                stderr="",
            )

    worker = InstructionApplyRunner(settings, store, review, runner=FakeRunner())
    monkeypatch.setattr(
        "pipeline.server.instruction_worker.executable_path",
        lambda value: "/workspace/letools/bin/letools",
    )

    result = worker._worker("recover", "/workspace/dataset")
    assert result == {"operation_id": "apply-a"}
```

Add a fail-closed test by binding metadata with
`slurm_execution_id="expired"` to a fake broker returning `None` from
`session_for_execution()`. Assert the queued row becomes `FAILED`, its error
mentions re-login, and neither `subprocess.run()` nor local execution is called.

- [ ] **Step 2: Run the focused tests and confirm local execution remains**

```bash
pytest tests/test_instruction_api.py::test_instruction_apply_records_submitter tests/test_user_identity.py::test_instruction_worker_uses_bound_remote_command -v
```

Expected: FAIL because `enqueue()` has no principal, `InstructionApplyRunner`
has no JobRunner dependency, and `_worker()` calls local `subprocess.run()`.

- [ ] **Step 3: Inject JobRunner and bind every worker lifecycle**

Create the runner with the shared execution object:

```python
instruction_apply = InstructionApplyRunner(
    settings,
    ingest.store,
    instruction_review,
    runner=runner,
)
```

Change enqueue:

```python
def enqueue(
    self,
    dataset_id,
    body,
    episode_indices: set[int] | None = None,
    *,
    principal: Principal | None = None,
):
    submitter = self.runner.submission_metadata(principal)["submitted_as"]
    fingerprint = digest(
        {
            "dataset": key,
            "path": str(path),
            "source_signature": snapshot["signature"],
            "episode_indices": episode_scope,
            "assignments": assignments,
            "submitted_by": submitter,
        }
    )
    row = {
        "id": apply_id,
        "dataset_id": dataset_id,
        "dataset_key": key,
        "dataset_path": str(path),
        "source_signature": snapshot["signature"],
        "revision": draft["revision"],
        "episode_indices": episode_scope,
        "fingerprint": fingerprint,
        "spec_path": str(spec_path),
        "state": "QUEUED",
        "stage": "等待应用",
        "progress": "已冻结 Episode 指令与来源版本",
        "created_at": now(),
        **self.runner.submission_metadata(principal),
    }
```

Before queue insertion, require writable access:

```python
self.runner.require_principal_path_access(
    principal,
    path,
    writable=True,
)
```

Change the tick boundary:

```python
def tick(self):
    queued = [
        row
        for row in self.store.all("instruction_apply")
        if row.get("state") == "QUEUED"
    ]
    if not queued:
        return False
    row = queued[-1]
    try:
        with self.runner.bind_record_session(row):
            self.run(row)
    except DomainError as exc:
        row.update(
            state="INTERRUPTED",
            stage="等待提交人重新登录",
            progress="执行会话已失效，尚未继续恢复或写入",
            error=exc.message,
            finished_at=now(),
        )
        self.store.put("instruction_apply", row["id"], row)
    return True
```

Change `_worker()`:

```python
def _worker(self, *args):
    tool = executable_path(self.settings.letools_bin)
    python = Path(tool).with_name("python") if tool else None
    if python is None or not python.is_file():
        raise DomainError("应用语言指令需要 LeTools Python 环境", 503)
    try:
        result = self.runner.run_captured(
            [
                str(python),
                str(Path(__file__).with_name("instruction_writer_worker.py")),
                *args,
            ],
            timeout=self.settings.command_timeout,
            remote_env=None,
        )
    except subprocess.TimeoutExpired as exc:
        raise DomainError(
            "应用语言指令超时；事务将在下次启动时恢复",
            504,
        ) from exc
    if result.returncode:
        raise DomainError(
            "应用语言指令失败：" + result.stderr[-1600:],
            409,
        )
    try:
        return json.loads(result.stdout)
    except ValueError as exc:
        raise DomainError("语言指令写入器未返回有效结果", 502) from exc
```

The existing recover, apply, rollback, committed-receipt, and dataset-lock calls
remain inside the bound `tick()` scope. A service job has no execution ID, so
`bind_record_session()` yields `None` and `run_captured()` retains the local
service-account behavior.

- [ ] **Step 4: Run instruction, conflict, and identity suites**

```bash
pytest tests/test_instruction_api.py tests/test_instruction_review.py tests/test_instruction_writer.py tests/test_user_identity.py -q
```

Expected: PASS. In-place edits, rollback, receipt reconciliation, and the
manual-trim conflict still behave as before. Personal worker execution uses the
bound SSH context.

- [ ] **Step 5: Commit**

```bash
git add pipeline/server/app.py pipeline/server/instruction_worker.py tests/test_instruction_api.py tests/test_instruction_review.py tests/test_user_identity.py
git commit -m "feat(server): apply instructions as submitting user"
```

### Task 5: Execute episode scoring under the submitter

**Files:**
- Modify: `pipeline/server/episode_scores.py:1-155`
- Modify: `pipeline/server/viewer_gateway.py`
- Modify: `pipeline/server/app.py:217-250`
- Modify: `pipeline/server/app.py:900-955`
- Test: `tests/test_episode_scores.py`
- Test: `tests/test_user_identity.py`

**Interfaces:**
- Produces: `EpisodeScoreCache(settings, store, runner: JobRunner | None = None)`
- Produces: `EpisodeScoreCache.start(preview, *, principal: Principal | None = None) -> dict`
- Produces: personal score records with standard execution metadata
- Consumes: `JobRunner.bind_record_session()` and `run_captured()`

- [ ] **Step 1: Write the failing score identity test**

```python
def test_episode_score_uses_submitter_control_master(
    settings_factory, tmp_path
):
    settings = settings_factory("server", letools_bin=sys.executable)
    root = dataset(tmp_path / "dataset")
    store = Store(settings.root / "state")
    principal = Principal(
        username="alice",
        uid=1001,
        login_node="H800-node11",
        execution_id="execution-a",
        auth_method="h800-ssh",
    )
    report = {
        "episodes": [{"episode_index": 0, "overall_score": 100}],
        "total_episodes": 1,
    }

    class Runner:
        def __init__(self):
            self.metadata = None
            self.argv = None

        def require_principal_path_access(
            self, principal, path, *, writable=False
        ):
            assert principal.username == "alice"
            assert Path(path) == root

        @staticmethod
        def submission_metadata(principal):
            return principal.job_metadata()

        @contextmanager
        def bind_record_session(self, row):
            self.metadata = dict(row)
            yield object()

        def run_captured(self, argv, *, timeout, remote_env=None):
            self.argv = argv
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(report),
                stderr="",
            )

    runner = Runner()
    cache = EpisodeScoreCache(settings, store, runner=runner)
    try:
        queued = cache.start(
            {"dataset_path": str(root)},
            principal=principal,
        )
        result = wait_for_score(
            cache,
            {"dataset_path": str(root)},
            principal=principal,
        )
    finally:
        cache.close()

    assert queued["submitted_as"] == "alice"
    assert runner.metadata["slurm_user"] == "alice"
    assert runner.metadata["slurm_execution_id"] == "execution-a"
    assert runner.argv[1].endswith("episode_score_worker.py")
    assert result["state"] == "COMPLETED"
    assert result["submitted_as"] == "alice"
    assert "slurm_execution_id" not in result
```

Update the `wait_for_score()` helper in this test module to accept
`principal=None` and pass it through to `cache.status()`.

Add a deduplication boundary test:

```python
def test_episode_score_keys_are_per_submitter(settings_factory, tmp_path):
    settings = settings_factory("server")
    store = Store(settings.root / "state")
    root = dataset(tmp_path / "dataset")
    cache = EpisodeScoreCache(settings, store)
    alice = Principal(
        username="alice",
        uid=1001,
        auth_method="h800-ssh",
        execution_id="execution-a",
    )
    bob = Principal(
        username="bob",
        uid=1002,
        auth_method="h800-ssh",
        execution_id="execution-b",
    )
    try:
        alice_key = cache._identity({"dataset_path": str(root)}, alice)[1]
        bob_key = cache._identity({"dataset_path": str(root)}, bob)[1]
        service_key = cache._identity({"dataset_path": str(root)}, None)[1]
    finally:
        cache.close()

    assert len({alice_key, bob_key, service_key}) == 3
```

Add an embedded API test that monkeypatches
`app.state.viewer_transport.scores.start` and asserts it receives the browser
Principal:

```python
def test_embedded_score_endpoint_passes_principal(
    settings_factory, tmp_path, monkeypatch
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
    app.state.viewer.registry.register(
        "job-a", dataset(tmp_path / "dataset")
    )
    monkeypatch.setattr(
        app.state.user_sessions,
        "principal",
        lambda cookie: Principal(
            username="alice",
            uid=1001,
            login_node="H800-node11",
            execution_id="execution-a",
            auth_method="h800-ssh",
        ),
    )
    captured = []
    monkeypatch.setattr(
        app.state.viewer_transport.scores,
        "start",
        lambda preview, *, principal: captured.append(principal)
        or {"state": "QUEUED"},
    )

    with TestClient(app) as client:
        client.cookies.set(
            "pipeline_server", "opaque-user-cookie", domain="testserver"
        )
        response = client.post("/api/episode-scores/job-a")

    assert response.status_code == 202
    assert captured[0].username == "alice"
```

- [ ] **Step 2: Run the tests and confirm scoring stays local**

```bash
pytest \
  tests/test_episode_scores.py::test_episode_score_uses_submitter_control_master \
  tests/test_episode_scores.py::test_episode_score_keys_are_per_submitter \
  tests/test_episode_scores.py::test_embedded_score_endpoint_passes_principal -v
```

Expected: FAIL because `EpisodeScoreCache` has no runner argument and
score keys are not submitter-scoped, and the score routes do not accept or
forward a Principal.

- [ ] **Step 3: Bind the score worker**

Add the runner dependency:

```python
class EpisodeScoreCache:
    def __init__(self, settings, store, runner=None):
        self.settings = settings
        self.store = store
        self.runner = runner
        # Existing pool, locks, pending, and stop state stay unchanged.

    def _identity(self, preview, principal=None):
        root = Path(preview["dataset_path"])
        submitter = (
            principal.username
            if (
                principal is not None
                and not principal.is_admin
                and principal.auth_method == "h800-ssh"
            )
            else "service-account"
        )
        key = hashlib.sha256(
            json.dumps(
                {"path": str(root), "submitted_as": submitter},
                sort_keys=True,
            ).encode()
        ).hexdigest()
        return root, key, dataset_signature(root)
```

Persist identity before queueing:

```python
def start(self, preview, *, principal=None):
    with self.lock:
        root, key, signature = self._identity(preview, principal)
        current = self.status(preview, principal=principal)
        if current["state"] in {"QUEUED", "RUNNING", "COMPLETED"}:
            return current
        if key in self.pending and not self.pending[key].done():
            return {"state": "RUNNING"}
        tool = executable_path(self.settings.letools_bin)
        python = Path(tool).with_name("python") if tool else None
        if python is None or not python.is_file() or not os.access(python, os.X_OK):
            raise DomainError("未找到 LeTools Python，无法运行 Episode 评分", 503)
        if self.runner is not None:
            self.runner.require_principal_path_access(principal, root)
            metadata = self.runner.submission_metadata(principal)
        elif principal is not None:
            metadata = principal.job_metadata()
        else:
            metadata = {
                "owner_user_id": None,
                "owner_uid": None,
                "submitted_as": "service-account",
                "slurm_user": None,
                "slurm_execution_id": None,
                "submission_auth": "internal",
            }
        result = {
            "state": "QUEUED",
            "signature": signature,
            "created_at": time.time(),
            **metadata,
        }
        self.store.put("episode-score", key, result)
        self.pending[key] = self.pool.submit(
            self._run,
            root,
            key,
            signature,
            str(python),
            metadata,
        )
        return public_execution_record(result)
```

Run through the bound execution context:

```python
def _run(self, root, key, signature, python, metadata):
    result = {
        "state": "RUNNING",
        "signature": signature,
        "created_at": time.time(),
        **metadata,
    }
    self.store.put("episode-score", key, result)
    try:
        worker = Path(__file__).with_name("episode_score_worker.py")
        if self.runner is None:
            completed = subprocess.run(
                [python, str(worker), str(root)],
                capture_output=True,
                text=True,
                timeout=self.settings.command_timeout,
                env=self.settings.tool_environment(),
            )
        else:
            with self.runner.bind_record_session(metadata):
                completed = self.runner.run_captured(
                    [python, str(worker), str(root)],
                    timeout=self.settings.command_timeout,
                )
        if completed.returncode:
            raise ValueError(completed.stderr[-1200:] or "评分程序执行失败")
        report = json.loads(completed.stdout)
        json.dumps(report, allow_nan=False)
        rows = report["episodes"]
        total = report["total_episodes"]
        if len(rows) != total or [
            row["episode_index"] for row in rows
        ] != list(range(total)):
            raise ValueError("评分结果未覆盖全部 Episode")
        if dataset_signature(root) != signature:
            raise ValueError("评分期间数据集发生变化，请重新计算")
        result.update(state="COMPLETED", report=report)
    except Exception as exc:
        result.update(state="FAILED", error=str(exc))
    result["finished_at"] = time.time()
    self.store.put("episode-score", key, result)
```

Make `status(preview, *, principal=None)` compute the same submitter-scoped key
and return `public_execution_record(result)`. In embedded mode, construct
`ViewerTransport(..., runner=runner)` and call
`scores.start(row, principal=principal(request))`.

Thread the authenticated Principal through both native score routes:

```python
@app.get("/api/episode-scores/{dataset_id}")
def embedded_score_status(dataset_id: str, request: Request):
    row = registered_preview(viewer_transport, dataset_id)
    result = viewer_transport.scores.status(
        row, principal=principal(request)
    )
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


@app.post("/api/episode-scores/{dataset_id}", status_code=202)
def embedded_score_start(dataset_id: str, request: Request):
    row = registered_preview(viewer_transport, dataset_id)
    result = viewer_transport.scores.start(
        row, principal=principal(request)
    )
    return JSONResponse(
        result, status_code=202, headers={"Cache-Control": "no-store"}
    )
```

The standalone `create_gateway()` constructs `ViewerTransport` without a
runner, preserving service-token compatibility while 8082 remains online.

- [ ] **Step 4: Run score and identity tests**

```bash
pytest tests/test_episode_scores.py tests/test_user_identity.py -q
```

Expected: PASS. Embedded scoring uses the submitter's ControlMaster; standalone
8082 and machine-token score requests remain service-account operations.

- [ ] **Step 5: Commit**

```bash
git add pipeline/server/episode_scores.py pipeline/server/viewer_gateway.py pipeline/server/app.py tests/test_episode_scores.py tests/test_user_identity.py
git commit -m "feat(server): score episodes as submitting user"
```

### Task 6: Prove end-to-end identity through the embedded Viewer APIs

**Files:**
- Modify: `pipeline/server/app.py:900-955`
- Modify: `pipeline/server/subtask_export.py:580-690`
- Modify: `tests/test_instruction_api.py`
- Modify: `tests/test_subtask_review.py`
- Modify: `tests/test_stationary_trim.py`
- Modify: `tests/test_episode_scores.py`
- Modify: `tests/test_user_identity.py`

**Interfaces:**
- Consumes: `pipeline_server` browser cookie
- Produces: instruction apply, manual trim, review export, and episode score all
  receive the same `request.state.principal`
- Produces: admin bearer requests remain service-account tasks

- [ ] **Step 1: Write the failing instruction endpoint identity test**

```python
def test_embedded_instruction_apply_passes_browser_principal(
    settings_factory, monkeypatch
):
    monkeypatch.setattr(
        SlurmIdentityBroker,
        "principal",
        lambda self, cookie: Principal(
            username="alice",
            uid=1001,
            login_node="H800-node11",
            execution_id="execution-a",
            auth_method="h800-ssh",
        )
        if cookie == "opaque-user-cookie"
        else None,
    )
    settings = settings_factory(
        "server",
        user_sessions={
            "enabled": True,
            "login_nodes": ["H800-node11"],
        },
    )
    app = create_app(settings, background=False)
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
    client = TestClient(app)
    client.cookies.set(
        "pipeline_server", "opaque-user-cookie", domain="testserver"
    )
    response = client.post(
        "/api/instructions/source/apply",
        json={
            "revision": 0,
            "source_signature": "signature-a",
        },
        headers={"Origin": "http://testserver"},
    )

    assert response.status_code == 202
    assert captured[0]["principal"].username == "alice"
    assert captured[0]["episode_indices"] is None
```

Add one test that exercises the other compute routes:

```python
def test_other_viewer_compute_routes_forward_principal(
    settings_factory, tmp_path, monkeypatch
):
    user = Principal(
        username="alice",
        uid=1001,
        login_node="H800-node11",
        execution_id="execution-a",
        auth_method="h800-ssh",
    )
    monkeypatch.setattr(
        SlurmIdentityBroker,
        "principal",
        lambda self, cookie: user if cookie == "opaque-user-cookie" else None,
    )
    settings = settings_factory(
        "server",
        user_sessions={
            "enabled": True,
            "login_nodes": ["H800-node11"],
        },
        viewer_gateway={"embedded": True},
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
        return {"id": "trim-a", "operation": "manual_trim"}

    def review_enqueue(dataset_id, body, *, principal):
        seen["review"] = (dataset_id, principal)
        return {"id": "export-a", "state": "QUEUED"}

    def score_start(preview, *, principal):
        seen["score"] = (preview["job_id"], principal)
        return {"state": "QUEUED"}

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
    client = TestClient(app)
    client.cookies.set(
        "pipeline_server", "opaque-user-cookie", domain="testserver"
    )
    origin = {"Origin": "http://testserver"}

    assert client.post(
        "/api/instructions/source/episodes/0/apply",
        json={"revision": 0, "source_signature": "signature-a"},
        headers=origin,
    ).status_code == 202
    assert client.post(
        "/api/manual-trims/source",
        json={"episode_index": 0, "remove_ranges": [[1, 2]]},
        headers=origin,
    ).status_code == 202
    assert client.post(
        "/api/subtask-review/source/exports",
        json={"revision": 0, "source_signature": "signature-a"},
        headers=origin,
    ).status_code == 202
    assert client.post(
        "/api/episode-scores/source",
        headers=origin,
    ).status_code == 202
    assert seen["episode"][1] == {0}
    assert seen["episode"][2].username == "alice"
    assert seen["manual"][1].username == "alice"
    assert seen["review"][1].username == "alice"
    assert seen["score"][1].username == "alice"
```

Add a machine-token regression:

```python
def test_embedded_compute_apis_keep_machine_token_as_service_account(
    settings_factory, monkeypatch
):
    settings = settings_factory("server")
    app = create_app(settings, background=False)
    captured = []

    def enqueue(dataset_id, body, episode_indices=None, *, principal):
        captured.append(principal)
        return {"id": "apply-a", "state": "QUEUED"}

    monkeypatch.setattr(app.state.instruction_apply, "enqueue", enqueue)
    with TestClient(app) as client:
        response = client.post(
            "/api/instructions/source/apply",
            json={
                "revision": 0,
                "source_signature": "signature-a",
            },
            headers={
                "Authorization": "Bearer "
                + app.state.runner.settings.token
            },
        )
    assert response.status_code == 202
    assert captured[0].is_admin is True
    assert captured[0].username == "service-account"
```

- [ ] **Step 2: Run the endpoint identity tests and confirm failures**

```bash
pytest tests/test_instruction_api.py::test_embedded_instruction_apply_passes_browser_principal tests/test_instruction_api.py::test_embedded_compute_apis_keep_machine_token_as_service_account -v
```

Expected: FAIL because instruction apply endpoints do not accept `Request` or
forward `principal(request)`.

- [ ] **Step 3: Thread Principal through the remaining Server routes**

Update both instruction apply routes:

```python
@app.post(
    "/api/instructions/{dataset_id}/episodes/{episode}/apply",
    status_code=202,
)
def instruction_episode_apply(
    dataset_id: str,
    episode: int,
    body: ApplyInstructionsRequest,
    request: Request,
):
    return instruction_apply.enqueue(
        dataset_id,
        body,
        episode_indices={episode},
        principal=principal(request),
    )


@app.post("/api/instructions/{dataset_id}/apply", status_code=202)
def instruction_apply_start(
    dataset_id: str,
    body: ApplyInstructionsRequest,
    request: Request,
):
    return instruction_apply.enqueue(
        dataset_id,
        body,
        principal=principal(request),
    )
```

Keep the existing review export call:

```python
job = runner.subtask_exports.enqueue_review(
    dataset_id,
    body,
    principal=principal(request),
)
```

The review export source already persists
`JobRunner.submission_metadata(principal)`. Change its deduplication key to use
`self.runner.submission_metadata(principal)["submitted_as"]` as well, so
device-token principals share the service-account key instead of creating a
device-owned personal key. Add a regression assertion for the resulting
`submitted_as="service-account"` record.

```python
key = "subtask-review:" + digest(
    {
        "source": str(source),
        "plan": plan["digest"],
        "name": output_name,
        "submitted_by": self.runner.submission_metadata(principal)[
            "submitted_as"
        ],
    }
)
```

No route accepts `username`, `uid`, `login_node`, or `execution_id` from the
request body.

- [ ] **Step 4: Run all embedded action suites**

```bash
pytest tests/test_instruction_api.py tests/test_instruction_review.py tests/test_stationary_trim.py tests/test_subtask_review.py tests/test_episode_scores.py tests/test_user_identity.py -q
```

Expected: PASS. Browser requests carry one consistent Principal across all four
compute actions. Machine-token requests remain service-account operations.

- [ ] **Step 5: Commit**

```bash
git add pipeline/server/app.py pipeline/server/subtask_export.py tests/test_instruction_api.py tests/test_subtask_review.py tests/test_stationary_trim.py tests/test_episode_scores.py tests/test_user_identity.py
git commit -m "test(server): cover viewer task identity end to end"
```

### Task 7: Gate identity activation independently from code deployment

**Files:**
- Create: `pipeline/server/execution_identity.py`
- Modify: `pipeline/shared/config.py:88-170`
- Modify: `pipeline/shared/config.py:180-235`
- Modify: `pipeline/server/app.py:235-275`
- Modify: `pipeline/server/app.py:770-955`
- Test: `tests/test_user_identity.py`
- Test: `tests/test_instruction_api.py`
- Test: `tests/test_stationary_trim.py`
- Test: `tests/test_subtask_review.py`
- Test: `tests/test_episode_scores.py`

**Interfaces:**
- Produces: `Settings.viewer_execution_identity: dict`
- Produces: `viewer_identity_enabled(settings) -> bool`
- Produces: `execution_principal(settings, request) -> Principal | None`
- Consumes: `viewer_execution_identity.enabled`

- [ ] **Step 1: Write the failing gate tests**

```python
def test_viewer_execution_identity_configuration_is_strict(
    settings_factory,
):
    assert settings_factory("server").viewer_execution_identity == {}
    enabled = settings_factory(
        "server",
        viewer_execution_identity={"enabled": True},
    )
    assert enabled.viewer_execution_identity == {"enabled": True}

    with pytest.raises(ValueError, match="viewer_execution_identity"):
        settings_factory(
            "server",
            viewer_execution_identity={"enabled": "yes"},
        )
    with pytest.raises(ValueError, match="viewer_execution_identity"):
        settings_factory(
            "server",
            viewer_execution_identity={"unknown": True},
        )
```

```python
def test_disabled_identity_gate_attributes_new_task_to_service(
    settings_factory, monkeypatch
):
    monkeypatch.setattr(
        SlurmIdentityBroker,
        "principal",
        lambda self, cookie: Principal(
            username="alice",
            uid=1001,
            login_node="H800-node11",
            execution_id="execution-a",
            auth_method="h800-ssh",
        )
        if cookie == "opaque-user-cookie"
        else None,
    )
    settings = settings_factory(
        "server",
        viewer_execution_identity={"enabled": False},
        user_sessions={
            "enabled": True,
            "login_nodes": ["H800-node11"],
        },
    )
    app = create_app(settings, background=False)
    captured = []
    monkeypatch.setattr(
        app.state.instruction_apply,
        "enqueue",
        lambda *args, principal, **kwargs: captured.append(principal)
        or {"id": "apply-a", "state": "QUEUED"},
    )
    with TestClient(app) as client:
        client.cookies.set(
            "pipeline_server", "opaque-user-cookie", domain="testserver"
        )
        response = client.post(
            "/api/instructions/source/apply",
            json={
                "revision": 0,
                "source_signature": "signature-a",
            },
            headers={"Origin": "http://testserver"},
        )
    assert response.status_code == 202
    assert captured == [None]
```

- [ ] **Step 2: Run the tests and confirm the gate is absent**

```bash
pytest tests/test_user_identity.py::test_viewer_execution_identity_configuration_is_strict tests/test_user_identity.py::test_disabled_identity_gate_attributes_new_task_to_service -v
```

Expected: FAIL because the setting and helper do not exist.

- [ ] **Step 3: Add the strict configuration and central gate**

```python
# pipeline/shared/config.py
viewer_execution_identity: dict = field(default_factory=dict)


def _validate_viewer_execution_identity_configuration(self):
    if not isinstance(self.viewer_execution_identity, dict):
        raise ValueError("viewer_execution_identity 必须是对象")
    unknown = set(self.viewer_execution_identity) - {"enabled"}
    if unknown:
        raise ValueError(
            "viewer_execution_identity 包含未知字段："
            + ", ".join(sorted(unknown))
        )
    if (
        "enabled" in self.viewer_execution_identity
        and type(self.viewer_execution_identity["enabled"]) is not bool
    ):
        raise ValueError("viewer_execution_identity.enabled 必须为布尔值")
```

Call `_validate_viewer_execution_identity_configuration()` from
`Settings.__post_init__()`, alongside the gateway validation from the first
plan.

```python
# pipeline/server/execution_identity.py
def viewer_identity_enabled(settings) -> bool:
    config = getattr(settings, "viewer_execution_identity", {})
    return isinstance(config, dict) and config.get("enabled") is True


def execution_principal(settings, request):
    principal = getattr(request.state, "principal", None)
    return principal if viewer_identity_enabled(settings) else None
```

For every new compute submission, replace
`principal=principal(request)` with:

```python
principal=execution_principal(settings, request)
```

Apply this to manual trim, instruction apply, and review export enqueue calls,
and to both episode score status and score start calls so reads and writes use
the same submitter-scoped cache key. Continue using `principal(request)` for
access checks and other read-only authenticated operations.

Update these browser identity tests to add
`viewer_execution_identity={"enabled": True}` to their `settings_factory`
call:

- `tests/test_instruction_api.py::test_embedded_instruction_apply_passes_browser_principal`
- `tests/test_instruction_api.py::test_other_viewer_compute_routes_forward_principal`
- `tests/test_episode_scores.py::test_embedded_score_endpoint_passes_principal`

Their direct manager tests continue passing a Principal explicitly and assert
the same storage contract.

This gate governs new submissions only. `JobRunner.bind_record_session()` and
`InstructionApplyRunner.tick()` continue reading the persisted row's
`slurm_execution_id`. Turning the gate off must not make an existing personal
job run as `service-account`.

- [ ] **Step 4: Run enabled, disabled, machine-token, and legacy tests**

```bash
pytest tests/test_user_identity.py tests/test_instruction_api.py tests/test_stationary_trim.py tests/test_subtask_review.py tests/test_episode_scores.py -q
```

Expected: PASS. Enabled non-admin requests create personal metadata; disabled
requests and admin machine tokens create service-account metadata; persisted
personal rows remain row-bound.

- [ ] **Step 5: Commit**

```bash
git add pipeline/shared/config.py pipeline/server/execution_identity.py pipeline/server/app.py tests/test_user_identity.py tests/test_instruction_api.py tests/test_stationary_trim.py tests/test_subtask_review.py tests/test_episode_scores.py
git commit -m "feat(server): gate viewer execution identity"
```

### Task 8: Verify failure recovery, permissions, and the production rollout

**Files:**
- Create: `docs/operations/viewer-user-execution-cutover.md`
- Modify: `configs/server.deployed.example.json`
- Modify: `configs/server.json`
- Modify: `tests/test_user_identity.py`
- Modify: `tests/test_instruction_review.py`
- Modify: `tests/test_stationary_trim.py`
- Modify: `tests/test_subtask_review.py`

**Interfaces:**
- Consumes: `viewer_execution_identity.enabled`
- Produces: fail-closed recovery behavior for expired personal sessions
- Produces: an enable, smoke, and rollback procedure for H800

- [ ] **Step 1: Write the failing expired-session tests**

```python
def test_expired_instruction_session_never_runs_locally(
    settings_factory, monkeypatch
):
    settings = settings_factory(
        "server",
        viewer_execution_identity={"enabled": True},
    )
    app = create_app(settings, background=False)
    app.state.runner.store.put(
        "instruction_apply",
        "apply-a",
        {
            "id": "apply-a",
            "dataset_id": "source",
            "dataset_key": "key",
            "dataset_path": str(settings.root / "source"),
            "source_signature": {},
            "revision": 0,
            "episode_indices": None,
            "fingerprint": "fingerprint-a",
            "spec_path": str(settings.root / "spec.json"),
            "state": "QUEUED",
            "stage": "等待应用",
            "progress": "",
            "created_at": 1,
            "owner_user_id": "alice",
            "owner_uid": 1001,
            "submitted_as": "alice",
            "slurm_user": "alice",
            "slurm_execution_id": "expired",
            "submission_auth": "h800-ssh",
        },
    )
    local = []
    monkeypatch.setattr(
        "pipeline.server.instruction_worker.subprocess.run",
        lambda *args, **kwargs: local.append((args, kwargs)),
    )

    app.state.instruction_apply.tick()
    row = app.state.runner.store.get("instruction_apply", "apply-a")

    assert row["state"] == "INTERRUPTED"
    assert "重新登录" in row["error"]
    assert local == []
```

```python
def test_expired_queued_manual_trim_fails_closed(
    settings_factory, monkeypatch
):
    settings = settings_factory("server", letools_bin=sys.executable)
    app = create_app(settings, background=False)
    runner = app.state.runner
    runner.store.put(
        "job",
        "trim-a",
        {
            "id": "trim-a",
            "state": "QUEUED",
            "operation": "manual_trim",
            "created_at": 1,
            "owner_user_id": "alice",
            "owner_uid": 1001,
            "submitted_as": "alice",
            "slurm_user": "alice",
            "slurm_execution_id": "expired",
            "submission_auth": "h800-ssh",
        },
    )
    runner.recover()
    row = runner.store.get("job", "trim-a")

    assert row["state"] == "INTERRUPTED"
    assert "用户会话失效" in row["error"]
```

Add a permission-denial test that expects 403 and no queued job when the remote
`test -w` probe returns a nonzero code. Add a legacy test that supplies no
`slurm_execution_id` and confirms the existing service-account execution path
still succeeds.

- [ ] **Step 2: Run the tests and confirm unsafe state handling**

```bash
pytest tests/test_user_identity.py::test_expired_instruction_session_never_runs_locally tests/test_user_identity.py::test_expired_queued_manual_trim_fails_closed -v
```

Expected before the final fix: the manual trim may already be interrupted, while
the instruction apply row retains a state that suggests recovery is in progress.

- [ ] **Step 3: Make recovery state explicit and fail closed**

Use this instruction binding error branch in `InstructionApplyRunner.tick()`:

```python
except DomainError as exc:
    row.update(
        state="INTERRUPTED",
        stage="等待提交人重新登录",
        progress="执行会话已失效，尚未继续恢复或写入",
        error=(
            "H800 登录会话已失效；请重新登录后重新提交。"
            "重新提交会先执行事务恢复，再开始新的写入。"
        ),
        finished_at=now(),
    )
    self.store.put("instruction_apply", row["id"], row)
```

Do not mark the dataset unchanged when the previous worker may have written a
journal. The new resubmission runs `instruction_writer_worker.py recover` under
the newly authenticated user before applying any new assignment.

Keep `JobRunner.recover()` behavior for queued personal jobs: they become
`INTERRUPTED` because the in-memory broker session cannot be reconstructed after
a process restart. Legacy rows without `slurm_execution_id` remain eligible for
service-account recovery.

Fix permission denial before enqueue by propagating the 403 from
`require_principal_path_access()`. Do not insert a job row when the probe fails.

- [ ] **Step 4: Add disabled defaults and the operational runbook**

Add to both Server configuration files:

```json
"viewer_execution_identity": {
  "enabled": false
}
```

The runbook uses this sequence:

```text
1. Confirm viewer_gateway.embedded=true and the workbench smoke test passes.
2. Confirm user_sessions.enabled=true.
3. Deploy this release with viewer_execution_identity.enabled=false.
4. Verify a machine-token task still records service-account metadata.
5. Set viewer_execution_identity.enabled=true.
6. Restart the Server release process once.
7. Log in with a non-admin H800 test account.
8. Run one small manual trim, one instruction apply on a disposable dataset,
   one review export, and one episode score.
9. Confirm each new task records the test username, UID, login node, and
   non-empty execution handle; confirm the API response omits the handle.
10. Confirm `squeue -u <test-user>` shows the workload under that user.
11. Confirm the changed files are readable through the existing publication
    step and the Server can update queue state.
12. Roll the UI or account audience only after the real-data fixture passes.
```

The rollback procedure is:

```text
1. Set viewer_execution_identity.enabled=false.
2. Leave the new release running.
3. Restart the Server release process once.
4. Verify new machine-token and browser tasks return to service-account mode.
5. Do not code-rollback while personal QUEUED or RUNNING rows remain.
6. For each personal queued row, allow the user session to finish or mark the
   row INTERRUPTED using the normal queue recovery procedure.
7. A code rollback is permitted only after no personal active rows remain.
8. Keep datasets and journals in place; do not attempt an automatic global
   rollback.
```

The runbook records the release identifier, config digest, test username, task
IDs, submitter metadata, Slurm user evidence, API redaction evidence, and
rollback owner. It does not record passwords, cookies, ControlPath values, or
execution handles.

- [ ] **Step 5: Run the complete execution and platform regression matrix**

```bash
pytest \
  tests/test_user_identity.py \
  tests/test_login_password.py \
  tests/test_instruction_api.py \
  tests/test_instruction_review.py \
  tests/test_instruction_writer.py \
  tests/test_stationary_trim.py \
  tests/test_subtask_review.py \
  tests/test_episode_scores.py \
  tests/test_viewer.py \
  tests/test_server_api.py \
  tests/test_workbench_lists.py \
  tests/test_ingest.py \
  tests/test_collections.py -q
ruff check \
  pipeline/shared/config.py \
  pipeline/shared/identity.py \
  pipeline/server/app.py \
  pipeline/server/execution_identity.py \
  pipeline/server/jobs.py \
  pipeline/server/trims.py \
  pipeline/server/instruction_worker.py \
  pipeline/server/episode_scores.py \
  pipeline/server/viewer_gateway.py \
  tests/test_user_identity.py
rg -n "slurm_execution_id" pipeline/web pipeline/server/viewer_ui tests --glob '*.{js,ts,tsx,html}'
```

Expected: all Python tests pass, Ruff reports no issues, and the final search
finds no browser-facing use of the private execution field. Any optional
LeTools tests that skip locally must be run with the production LeTools fixture
before enabling `viewer_execution_identity`.

- [ ] **Step 6: Commit**

```bash
git add \
  configs/server.json \
  configs/server.deployed.example.json \
  docs/operations/viewer-user-execution-cutover.md \
  pipeline/server/instruction_worker.py \
  tests/test_user_identity.py \
  tests/test_instruction_review.py \
  tests/test_stationary_trim.py \
  tests/test_subtask_review.py
git commit -m "docs(deploy): add viewer user execution cutover"
```
