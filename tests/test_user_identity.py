import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from pipeline.server.app import create_app
from pipeline.server.instruction_worker import InstructionApplyRunner
from pipeline.server.jobs import JobRunner
from pipeline.server.user_sessions import SlurmIdentityBroker, UserSession
from pipeline.shared.core import DomainError, Store, now
from pipeline.shared.identity import Principal, require_task_access


def executable(path: Path, source: str) -> Path:
    path.write_text(source)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def fake_ssh_commands(tmp_path):
    password_probe = tmp_path / "password.txt"
    argv_probe = tmp_path / "sshpass-argv.json"
    env_probe = tmp_path / "env.json"
    ssh = executable(
        tmp_path / "ssh",
        """#!/usr/bin/env python3
import os
import sys

if "-O" in sys.argv and "exit" in sys.argv:
    for value in sys.argv:
        if value.startswith("ControlPath="):
            try:
                os.unlink(value.split("=", 1)[1])
            except FileNotFoundError:
                pass
    raise SystemExit(0)
print("1001")
print(os.environ.get("FAKE_HOSTNAME", "H800-node11"))
""",
    )
    sshpass = executable(
        tmp_path / "sshpass",
        f"""#!/usr/bin/env python3
import json
import pathlib
import sys

password = sys.stdin.buffer.read()
pathlib.Path({str(password_probe)!r}).write_bytes(password)
pathlib.Path({str(argv_probe)!r}).write_text(json.dumps(sys.argv))
pathlib.Path({str(env_probe)!r}).write_text(json.dumps(dict(__import__("os").environ)))
for value in sys.argv:
    if value.startswith("ControlPath="):
        pathlib.Path(value.split("=", 1)[1]).touch(mode=0o600)
""",
    )
    return ssh, sshpass, password_probe, argv_probe, env_probe


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


def test_disabled_identity_gate_attributes_new_task_to_service(
    settings_factory, monkeypatch
):
    settings = settings_factory(
        "server",
        viewer_execution_identity={"enabled": False},
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

    def enqueue(*args, principal, **kwargs):
        captured.append(principal)
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
    assert captured == [None]


def test_h800_login_keeps_password_out_of_argv_and_session_state(
    settings_factory, tmp_path
):
    ssh, sshpass, password_probe, argv_probe, env_probe = fake_ssh_commands(tmp_path)
    settings = settings_factory(
        "server",
        user_sessions={
            "enabled": True,
            "login_nodes": ["H800-node11"],
            "ssh_bin": str(ssh),
            "sshpass_bin": str(sshpass),
            "connect_host": "127.0.0.1",
            "control_dir": f"/tmp/bodypipeline-{tmp_path.name}",
        },
    )
    broker = SlurmIdentityBroker(settings)
    password = "not-written-anywhere"

    session = broker.login("alice", password, "H800-node11")
    principal = broker.principal(session.session_id)

    assert password_probe.read_text() == password
    assert password not in argv_probe.read_text()
    assert password not in env_probe.read_text()
    assert password not in repr(session)
    assert password not in repr(principal)
    assert principal.username == "alice"
    assert principal.uid == 1001
    assert principal.login_node == "H800-node11"
    assert principal.execution_id == session.execution_id

    broker.logout(session.session_id)
    assert broker.principal(session.session_id) is None


def test_h800_login_rejects_unapproved_node_and_unsafe_username(settings_factory):
    settings = settings_factory(
        "server",
        user_sessions={
            "enabled": True,
            "login_nodes": ["H800-node11"],
            "ssh_bin": "/usr/bin/ssh",
            "sshpass_bin": "/usr/bin/sshpass",
        },
    )
    broker = SlurmIdentityBroker(settings)
    with pytest.raises(DomainError, match="登录节点"):
        broker.login("alice", "secret", "untrusted-node")
    with pytest.raises(DomainError, match="用户名"):
        broker.login("alice; touch /tmp/pwned", "secret", "H800-node11")


def test_remote_command_uses_control_socket_not_password(settings_factory, tmp_path):
    ssh, sshpass, *_ = fake_ssh_commands(tmp_path)
    settings = settings_factory(
        "server",
        user_sessions={
            "enabled": True,
            "login_nodes": ["H800-node11"],
            "ssh_bin": str(ssh),
            "sshpass_bin": str(sshpass),
            "connect_host": "127.0.0.1",
            "control_dir": f"/tmp/bodypipeline-{tmp_path.name}",
        },
    )
    broker = SlurmIdentityBroker(settings)
    session = broker.login("alice", "secret", "H800-node11")

    argv = broker.command_argv(
        session,
        ["/usr/bin/srun", "--partition", "dev", "/opt/tool", "convert"],
        cwd=tmp_path,
        remote_env={"HF_HOME": "/models"},
    )

    assert argv[0] == str(ssh)
    assert str(session.control_path) in argv
    assert "BatchMode=yes" in argv
    assert "secret" not in " ".join(argv)
    assert "cd " in argv[-1]
    assert "umask 000" in argv[-1]
    assert "HF_HOME=/models" in argv[-1]
    assert "/usr/bin/srun --partition dev /opt/tool convert" in argv[-1]


def test_server_browser_login_issues_opaque_cookie_and_reports_identity(
    settings_factory, monkeypatch
):
    settings = settings_factory(
        "server",
        user_sessions={
            "enabled": True,
            "login_nodes": ["H800-node11"],
        },
    )

    def fake_login(self, username, password, login_node):
        assert password == "correct"
        session = UserSession(
            session_id="opaque-cookie-token",
            execution_id="private-execution-token",
            username=username,
            uid=1001,
            login_node=login_node,
            control_path=Path("/tmp/not-used"),
            expires_at=now() + 3600,
        )
        with self._lock:
            self._by_token[session.session_id] = session
            self._by_execution[session.execution_id] = session
        return session

    monkeypatch.setattr(SlurmIdentityBroker, "login", fake_login)
    with TestClient(create_app(settings, background=False)) as http:
        options = http.get("/api/auth-options").json()
        assert options["user_login"] is True
        assert options["login_nodes"] == ["H800-node11"]

        response = http.post(
            "/api/login",
            json={
                "username": "alice",
                "password": "correct",
                "login_node": "H800-node11",
            },
        )
        assert response.status_code == 200
        cookie = response.cookies.get("pipeline_server")
        assert cookie == "opaque-cookie-token"
        assert "alice" not in cookie
        assert "private-execution-token" not in cookie

        config = http.get("/api/config").json()
        assert config["current_user"]["username"] == "alice"
        assert config["current_user"]["login_node"] == "H800-node11"
        assert config["current_user"]["is_admin"] is False

        assert http.post("/api/logout").status_code == 200
        assert http.get("/api/config").status_code == 401


def test_machine_token_remains_admin_when_user_login_is_enabled(settings_factory):
    settings = settings_factory(
        "server",
        user_sessions={
            "enabled": True,
            "login_nodes": ["H800-node11"],
        },
    )
    with TestClient(create_app(settings, background=False)) as http:
        config = http.get(
            "/api/config",
            headers={"Authorization": "Bearer " + settings.token},
        )
        assert config.status_code == 200
        assert config.json()["current_user"]["is_admin"] is True


def test_bearer_identity_takes_precedence_over_user_cookie(
    settings_factory, monkeypatch
):
    settings = settings_factory(
        "server",
        viewer_gateway={"embedded": True},
        user_sessions={
            "enabled": True,
            "login_nodes": ["H800-node11"],
        },
    )

    def fake_login(self, username, password, login_node):
        session = UserSession(
            session_id="user-cookie",
            execution_id="user-execution",
            username=username,
            uid=1001,
            login_node=login_node,
            control_path=Path("/tmp/user-cookie"),
            expires_at=now() + 3600,
        )
        with self._lock:
            self._by_token[session.session_id] = session
            self._by_execution[session.execution_id] = session
        return session

    monkeypatch.setattr(SlurmIdentityBroker, "login", fake_login)
    with TestClient(create_app(settings, background=False)) as http:
        login = http.post(
            "/api/login",
            json={
                "username": "alice",
                "password": "correct",
                "login_node": "H800-node11",
            },
        )
        assert login.status_code == 200

        cookie_config = http.get("/api/config")
        assert cookie_config.status_code == 200
        assert cookie_config.json()["current_user"]["username"] == "alice"
        assert "execution_id" not in cookie_config.json()["current_user"]
        assert "user-execution" not in cookie_config.text
        assert (
            cookie_config.json()["current_user"]["auth_method"]
            == "h800-ssh"
        )

        config = http.get(
            "/api/config",
            headers={"Authorization": "Bearer " + settings.token},
        )
        assert config.status_code == 200
        assert config.json()["current_user"] == {
            "username": "service-account",
            "uid": None,
            "login_node": None,
            "auth_method": "service-token",
            "is_admin": True,
        }


def test_authenticated_users_share_task_access():
    job = {
        "id": "job-owned-by-alice",
        "state": "COMPLETED",
        "dataset_name": "dataset",
        "owner_user_id": "alice",
        "submitted_as": "alice",
        "slurm_user": "alice",
    }

    require_task_access(job, Principal(username="bob", uid=1002))
    with pytest.raises(DomainError, match="登录"):
        require_task_access(job, None)


def test_principal_metadata_contains_owner_but_never_password():
    principal = Principal(
        username="alice",
        uid=1001,
        login_node="H800-node11",
        execution_id="private-execution-token",
        auth_method="h800-ssh",
    )
    assert principal.job_metadata() == {
        "owner_user_id": "alice",
        "owner_uid": 1001,
        "submitted_as": "alice",
        "slurm_user": "alice",
        "slurm_execution_id": "private-execution-token",
        "submission_auth": "h800-ssh",
    }


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


def test_instruction_worker_uses_bound_remote_command(
    settings_factory, tmp_path, monkeypatch
):
    settings = settings_factory("server")
    store = Store(settings.root / "state")
    review = SimpleNamespace()
    tool = executable(tmp_path / "letools", "")
    executable(tool.with_name("python"), "")

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

    worker = InstructionApplyRunner(
        settings, store, review, runner=FakeRunner()
    )
    monkeypatch.setattr(
        "pipeline.server.instruction_worker.executable_path",
        lambda value: str(tool),
    )

    result = worker._worker("recover", "/workspace/dataset")
    assert result == {"operation_id": "apply-a"}


def test_instruction_worker_interrupts_when_bound_session_expired(
    settings_factory, monkeypatch
):
    settings = settings_factory("server")
    store = Store(settings.root / "state")

    class ExpiredBroker:
        def session_for_execution(self, execution_id):
            assert execution_id == "expired"
            return None

    execution = JobRunner(
        settings,
        SimpleNamespace(store=store),
        user_sessions=ExpiredBroker(),
    )
    worker = InstructionApplyRunner(
        settings, store, SimpleNamespace(), runner=execution
    )
    store.put(
        "instruction_apply",
        "apply-expired",
        {
            "id": "apply-expired",
            "dataset_id": "source",
            "state": "QUEUED",
            "owner_user_id": "alice",
            "owner_uid": 1001,
            "submitted_as": "alice",
            "slurm_user": "alice",
            "slurm_execution_id": "expired",
            "submission_auth": "h800-ssh",
        },
    )
    worker_calls = []
    monkeypatch.setattr(
        worker, "run", lambda row: worker_calls.append(row)
    )
    local_calls = []
    monkeypatch.setattr(
        "pipeline.server.instruction_worker.subprocess.run",
        lambda *args, **kwargs: local_calls.append((args, kwargs)),
    )

    assert worker.tick() is True
    row = store.get("instruction_apply", "apply-expired")
    assert row["state"] == "INTERRUPTED"
    assert "重新登录" in row["error"]
    assert worker_calls == []
    assert local_calls == []


def test_instruction_startup_recovery_defers_personal_receipt_until_bound(
    settings_factory,
):
    settings = settings_factory("server")
    store = Store(settings.root / "state")
    dataset = settings.root / "dataset"
    meta = dataset / "meta"
    meta.mkdir(parents=True)
    apply_id = "apply-personal-receipt"
    journal = (
        meta / f".bodypipeline-instruction-journal-{apply_id}.json"
    )
    journal.write_text(
        json.dumps(
            {
                "state": "committed",
                "operation_id": apply_id,
                "report": {"updated_episodes": [0]},
            }
        )
    )
    stage = (
        meta / f".bodypipeline-instruction-stage-{apply_id}"
    )
    backup = (
        meta / f".bodypipeline-instruction-backup-{apply_id}"
    )
    stage.mkdir()
    backup.mkdir()
    (stage / "private-evidence").write_text("stage")
    (backup / "private-evidence").write_text("backup")
    spec = settings.root / "reports" / f"instruction-apply-{apply_id}.json"
    spec.parent.mkdir(parents=True)
    spec.write_text(
        json.dumps(
            {
                "operation_id": apply_id,
                "dataset_path": str(dataset),
                "assignments": [],
            }
        )
    )

    class ExpiredBroker:
        def session_for_execution(self, execution_id):
            assert execution_id == "expired-execution"
            return None

    execution = JobRunner(
        settings,
        SimpleNamespace(store=store),
        user_sessions=ExpiredBroker(),
    )
    finalized = []
    worker = InstructionApplyRunner(
        settings,
        store,
        SimpleNamespace(
            resolve=lambda dataset_id: None,
            finalize_apply=lambda *args: finalized.append(args),
        ),
        runner=execution,
    )
    store.put(
        "instruction_apply",
        apply_id,
        {
            "id": apply_id,
            "dataset_id": "source",
            "dataset_key": "dataset-key",
            "dataset_path": str(dataset),
            "source_signature": "a" * 64,
            "spec_path": str(spec),
            "state": "RUNNING",
            "owner_user_id": "alice",
            "owner_uid": 1001,
            "submitted_as": "alice",
            "slurm_user": "alice",
            "slurm_execution_id": "expired-execution",
            "submission_auth": "h800-ssh",
        },
    )

    worker.recover()

    recovered = store.get("instruction_apply", apply_id)
    assert recovered["state"] == "QUEUED"
    assert journal.is_file()
    assert stage.is_dir()
    assert backup.is_dir()
    assert "committed_report" not in json.loads(spec.read_text())
    assert finalized == []

    assert worker.tick() is True

    interrupted = store.get("instruction_apply", apply_id)
    assert interrupted["state"] == "INTERRUPTED"
    assert "重新登录" in interrupted["error"]
    assert journal.is_file()
    assert stage.is_dir()
    assert backup.is_dir()
    assert finalized == []


def test_instruction_worker_interrupts_when_session_expires_during_work(
    settings_factory, monkeypatch
):
    settings = settings_factory("server")
    store = Store(settings.root / "state")
    dataset = settings.root / "dataset"
    dataset.mkdir()

    class ActiveBroker:
        def session_for_execution(self, execution_id):
            return SimpleNamespace(username="alice", uid=1001)

    execution = JobRunner(
        settings,
        SimpleNamespace(store=store),
        user_sessions=ActiveBroker(),
    )
    worker = InstructionApplyRunner(
        settings,
        store,
        SimpleNamespace(resolve=lambda dataset_id: None),
        runner=execution,
    )
    row = {
        "id": "apply-expired-midflight",
        "dataset_id": "source",
        "dataset_key": "dataset-key",
        "dataset_path": str(dataset),
        "state": "QUEUED",
        "owner_user_id": "alice",
        "owner_uid": 1001,
        "submitted_as": "alice",
        "slurm_user": "alice",
        "slurm_execution_id": "execution-a",
        "submission_auth": "h800-ssh",
    }
    store.put("instruction_apply", row["id"], row)
    monkeypatch.setattr(
        worker, "_committed_receipt", lambda active: None
    )
    monkeypatch.setattr(
        worker,
        "_worker",
        lambda *args: (_ for _ in ()).throw(
            DomainError("任务发起者的 H800 登录会话已过期；请重新登录", 401)
        ),
    )

    assert worker.tick() is True
    persisted = store.get("instruction_apply", row["id"])
    assert persisted["state"] == "INTERRUPTED"
    assert "重新登录" in persisted["error"]


def test_instruction_worker_interrupts_when_rollback_session_expires(
    settings_factory, monkeypatch
):
    settings = settings_factory("server")
    store = Store(settings.root / "state")
    dataset = settings.root / "dataset"
    dataset.mkdir()

    class ActiveBroker:
        def session_for_execution(self, execution_id):
            return SimpleNamespace(username="alice", uid=1001)

    execution = JobRunner(
        settings,
        SimpleNamespace(store=store),
        user_sessions=ActiveBroker(),
    )
    worker = InstructionApplyRunner(
        settings,
        store,
        SimpleNamespace(resolve=lambda dataset_id: None),
        runner=execution,
    )
    row = {
        "id": "apply-expired-during-rollback",
        "dataset_id": "source",
        "dataset_key": "dataset-key",
        "dataset_path": str(dataset),
        "state": "QUEUED",
        "owner_user_id": "alice",
        "owner_uid": 1001,
        "submitted_as": "alice",
        "slurm_user": "alice",
        "slurm_execution_id": "execution-a",
        "submission_auth": "h800-ssh",
    }
    store.put("instruction_apply", row["id"], row)
    worker_calls = []

    def run_worker(*args):
        worker_calls.append(args)
        if len(worker_calls) == 1:
            return {}
        raise DomainError(
            "任务发起者的 H800 登录会话已过期；请重新登录", 401
        )

    monkeypatch.setattr(worker, "_committed_receipt", lambda active: None)
    monkeypatch.setattr(worker, "_worker", run_worker)
    monkeypatch.setattr(
        worker,
        "_frozen_assignments",
        lambda active: (_ for _ in ()).throw(
            DomainError("排队期间来源数据已变化", 409)
        ),
    )

    assert worker.tick() is True
    persisted = store.get("instruction_apply", row["id"])
    assert persisted["state"] == "INTERRUPTED"
    assert "重新登录" in persisted["error"]
    assert worker_calls == [
        ("recover", str(dataset)),
        ("recover", str(dataset)),
    ]


def test_instruction_worker_redacts_private_ssh_diagnostics(
    settings_factory, tmp_path, monkeypatch
):
    settings = settings_factory("server")
    store = Store(settings.root / "state")
    dataset = settings.root / "dataset"
    dataset.mkdir()

    class ActiveBroker:
        def session_for_execution(self, execution_id):
            return SimpleNamespace(username="alice", uid=1001)

    execution = JobRunner(
        settings,
        SimpleNamespace(store=store),
        user_sessions=ActiveBroker(),
    )
    control_path = (
        "/tmp/bodypipeline-501/ssh/alice-0123456789abcdef.sock"
    )
    execution_handle = "private-execution-handle"
    diagnostic = (
        "ssh: connect to socket "
        f"{control_path} failed; session={execution_handle}"
    )

    def run_captured(argv, *, timeout, remote_env=None):
        return SimpleNamespace(
            returncode=255,
            stdout="",
            stderr=diagnostic,
        )

    monkeypatch.setattr(execution, "run_captured", run_captured)
    tool = executable(tmp_path / "letools", "")
    executable(tool.with_name("python"), "")
    worker = InstructionApplyRunner(
        settings,
        store,
        SimpleNamespace(resolve=lambda dataset_id: None),
        runner=execution,
    )
    monkeypatch.setattr(
        "pipeline.server.instruction_worker.executable_path",
        lambda value: str(tool),
    )
    row = {
        "id": "apply-private-diagnostic",
        "dataset_id": "source",
        "dataset_key": "dataset-key",
        "dataset_path": str(dataset),
        "spec_path": str(settings.root / "missing-spec.json"),
        "state": "QUEUED",
        "owner_user_id": "alice",
        "owner_uid": 1001,
        "submitted_as": "alice",
        "slurm_user": "alice",
        "slurm_execution_id": "execution-a",
        "submission_auth": "h800-ssh",
    }
    store.put("instruction_apply", row["id"], row)

    assert worker.tick() is True

    stored = store.get("instruction_apply", row["id"])
    public = worker._public(stored)
    assert "应用语言指令失败" in stored["error"]
    assert "应用语言指令失败" in public["error"]
    for private in (control_path, execution_handle):
        assert private not in stored["error"]
        assert private not in json.dumps(public, ensure_ascii=False)


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
    app.state.runner.store.put(
        "check",
        "check-a",
        {
            "id": "check-a",
            "job_id": "job-a",
            "state": "QUEUED",
            "created_at": 1,
            "slurm_execution_id": "private-execution-token",
        },
    )
    app.state.runner.store.put(
        "analysis",
        "analysis-a",
        {
            "id": "analysis-a",
            "job_id": "job-a",
            "state": "QUEUED",
            "created_at": 1,
            "slurm_execution_id": "private-execution-token",
        },
    )
    app.state.runner.store.put(
        "annotation",
        "annotation-a",
        {
            "id": "annotation-a",
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
        checks = client.get("/api/jobs/job-a/checks")
        analyses = client.get("/api/jobs/job-a/analyses")
        analysis = client.get("/api/analyses/analysis-a")
        annotations = client.get("/api/annotations")
        annotation = client.get("/api/annotations/annotation-a")

    assert job.status_code == 200
    job_body = job.json()
    assert "slurm_execution_id" not in job_body
    assert job_body["owner_user_id"] == "alice"
    assert job_body["submitted_as"] == "alice"
    assert job_body["slurm_user"] == "alice"
    assert job_body["submission_auth"] == "h800-ssh"
    assert (
        app.state.runner.store.get("job", "job-a")["slurm_execution_id"]
        == "private-execution-token"
    )
    assert previews.status_code == 200
    assert "private-execution-token" not in previews.text
    for response in (checks, analyses, analysis, annotations, annotation):
        assert response.status_code == 200
        assert "slurm_execution_id" not in response.text
        assert "private-execution-token" not in response.text


def test_slurm_command_uses_the_submitting_users_control_master(
    settings_factory, tmp_path, monkeypatch
):
    ssh, sshpass, *_ = fake_ssh_commands(tmp_path)
    settings = settings_factory(
        "server",
        slurm={"enabled": True, "partition": "dev", "cpus_per_task": 4},
        user_sessions={
            "enabled": True,
            "login_nodes": ["H800-node11"],
            "ssh_bin": str(ssh),
            "sshpass_bin": str(sshpass),
            "connect_host": "127.0.0.1",
            "control_dir": f"/tmp/bodypipeline-{tmp_path.name}",
        },
    )
    broker = SlurmIdentityBroker(settings)
    session = broker.login("alice", "secret", "H800-node11")
    runner = JobRunner(
        settings,
        SimpleNamespace(store=Store(settings.root / "state")),
        user_sessions=broker,
    )
    captured = {}
    shared = []

    def popen(active, argv, **kwargs):
        captured.update(session=active, argv=argv, kwargs=kwargs)
        return SimpleNamespace(poll=lambda: 0, returncode=0)

    broker.popen = popen
    broker.run = lambda active, argv, **kwargs: shared.append((active, argv)) or SimpleNamespace(
        returncode=0
    )
    monkeypatch.setattr(
        "pipeline.server.jobs.executable_path",
        lambda name: f"/usr/bin/{name}",
    )
    work = tmp_path / "work"
    work.mkdir()

    with runner.bind_record_session(
        {
            "owner_user_id": "alice",
            "owner_uid": 1001,
            "submitted_as": "alice",
            "slurm_user": "alice",
            "slurm_execution_id": session.execution_id,
            "submission_auth": "h800-ssh",
        }
    ):
        code, _, _ = runner.command(["/opt/letools", "convert"], work, "convert")

    assert code == 0
    assert captured["session"].username == "alice"
    assert captured["argv"][0] == "/usr/bin/srun"
    assert captured["argv"][-2:] == ["/opt/letools", "convert"]
    assert captured["kwargs"]["cwd"] == work
    assert shared
    assert shared[-1][0] is session
    assert shared[-1][1][0].endswith("find")
    assert str(work) in shared[-1][1]
    assert shared[-1][1][2:5] == ["!", "-type", "l"]
