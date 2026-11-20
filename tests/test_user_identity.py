import stat
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from pipeline.server.app import create_app
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
        {"slurm_execution_id": session.execution_id}
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
