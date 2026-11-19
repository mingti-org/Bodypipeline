from __future__ import annotations

import os
import re
import secrets
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from pipeline.shared.core import DomainError
from pipeline.shared.identity import Principal

_USERNAME = re.compile(r"^[a-z_][a-z0-9_-]{0,31}$")
_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class UserSession:
    session_id: str
    execution_id: str
    username: str
    uid: int
    login_node: str
    control_path: Path
    expires_at: float
    connect_host: str = ""

    def principal(self) -> Principal:
        return Principal(
            username=self.username,
            uid=self.uid,
            login_node=self.login_node,
            execution_id=self.execution_id,
            auth_method="h800-ssh",
            is_admin=False,
        )


class SlurmIdentityBroker:
    """Authenticate H800 users and execute Slurm commands as those users.

    A password is used once to establish an OpenSSH ControlMaster. Only an
    opaque browser-session token and an internal execution handle are retained;
    the password itself is never written to state, logs, cookies, or argv.
    """

    def __init__(self, settings):
        self.settings = settings
        raw = getattr(settings, "user_sessions", None)
        self.config = raw if isinstance(raw, dict) else {}
        self.enabled = bool(self.config.get("enabled", False))
        self.login_nodes = tuple(
            str(value) for value in self.config.get("login_nodes", ["H800-node11"])
        )
        self.ssh = str(self.config.get("ssh_bin", "/usr/bin/ssh"))
        self.sshpass = str(self.config.get("sshpass_bin", "/usr/bin/sshpass"))
        self.connect_host = str(self.config.get("connect_host", "127.0.0.1"))
        raw_hosts = self.config.get("connect_hosts") or {}
        if not isinstance(raw_hosts, dict):
            raise ValueError("user_sessions.connect_hosts 必须为对象")
        self.connect_hosts = {
            str(node).strip(): str(host).strip()
            for node, host in raw_hosts.items()
            if str(node).strip() and str(host).strip()
        }
        self.port = int(self.config.get("port", 22))
        self.ttl = int(self.config.get("session_ttl_seconds", 24 * 3600))
        self.connect_timeout = int(self.config.get("connect_timeout_seconds", 20))
        self.strict_host_key_checking = str(
            self.config.get("strict_host_key_checking", "yes")
        )
        self.control_dir = self._control_directory()
        self._lock = threading.RLock()
        self._by_token: dict[str, UserSession] = {}
        self._by_execution: dict[str, UserSession] = {}

    def _control_directory(self) -> Path:
        configured = str(self.config.get("control_dir") or "").strip()
        if configured:
            path = Path(configured).expanduser().resolve()
        else:
            runtime = os.environ.get("XDG_RUNTIME_DIR")
            base = Path(runtime) if runtime else Path(tempfile.gettempdir())
            path = base / f"bodypipeline-{os.getuid()}" / "ssh"
        if len(str(path)) > 90:
            raise ValueError("user_sessions.control_dir 过长，OpenSSH socket 上限约 100 字符")
        try:
            path.mkdir(parents=True, mode=0o700, exist_ok=True)
            if path.is_symlink() or not path.is_dir():
                raise ValueError("user_sessions.control_dir 必须是非符号链接目录")
            path.chmod(0o700)
        except OSError as exc:
            raise ValueError(f"无法创建私有 SSH 会话目录：{path}") from exc
        return path

    def login(self, username, password, login_node) -> UserSession:
        if not self.enabled:
            raise DomainError("H800 用户登录未启用", 503)
        username = str(username or "").strip()
        login_node = str(login_node or "").strip()
        if not _USERNAME.fullmatch(username):
            raise DomainError("H800 用户名格式无效", 400)
        if login_node not in self.login_nodes:
            raise DomainError("登录节点不在允许列表中", 400)
        connect_host = self.connect_hosts.get(login_node, self.connect_host)
        if (
            not isinstance(password, str)
            or not password
            or len(password.encode("utf-8")) > 4096
        ):
            raise DomainError("H800 密码无效", 400)
        self.ssh = self._executable(self.ssh, "ssh")
        self.sshpass = self._executable(self.sshpass, "sshpass")

        session_id = secrets.token_urlsafe(32)
        execution_id = secrets.token_urlsafe(24)
        control_path = self.control_dir / (
            username[:16] + "-" + secrets.token_hex(8) + ".sock"
        )
        master = [
            self.sshpass,
            "-d",
            "0",
            self.ssh,
            "-M",
            "-N",
            "-f",
            "-o",
            "ControlMaster=yes",
            "-o",
            f"ControlPath={control_path}",
            "-o",
            f"ControlPersist={self.ttl}",
            "-o",
            f"ConnectTimeout={self.connect_timeout}",
            "-o",
            f"StrictHostKeyChecking={self.strict_host_key_checking}",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            "-p",
            str(self.port),
            "-l",
            username,
            connect_host,
        ]
        try:
            process = subprocess.Popen(
                master,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            process.communicate(
                password.encode("utf-8"), timeout=self.connect_timeout + 10
            )
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            control_path.unlink(missing_ok=True)
            raise DomainError("H800 SSH 登录超时", 504) from None
        except OSError:
            control_path.unlink(missing_ok=True)
            raise DomainError("无法启动 H800 SSH 登录", 503) from None
        if process.returncode:
            control_path.unlink(missing_ok=True)
            raise DomainError("H800 用户名、密码或登录节点不正确", 401)
        try:
            control_path.chmod(0o600)
        except OSError:
            pass

        probe = self._run_control(
            control_path,
            username,
            connect_host,
            "printf '%s\\n' \"$(id -u)\" \"$(hostname -s)\"",
        )
        lines = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
        if probe.returncode or len(lines) < 2 or not lines[0].isdecimal():
            self._exit_control(control_path, username, connect_host)
            raise DomainError(
                "H800 用户会话验证失败"
                + (
                    "：" + probe.stderr.strip()[-300:]
                    if probe.stderr
                    else ""
                ),
                401,
            )
        if lines[1] != login_node:
            self._exit_control(control_path, username, connect_host)
            raise DomainError(
                f"登录节点不匹配：请求 {login_node}，实际 {lines[1]}", 409
            )

        session = UserSession(
            session_id=session_id,
            execution_id=execution_id,
            username=username,
            uid=int(lines[0]),
            login_node=login_node,
            control_path=control_path,
            expires_at=time.time() + self.ttl,
            connect_host=connect_host,
        )
        with self._lock:
            self._cleanup_locked()
            self._by_token[session_id] = session
            self._by_execution[execution_id] = session
        return session

    @staticmethod
    def _executable(value, label: str) -> str:
        path = str(value or "").strip()
        found = shutil.which(path)
        if not found:
            raise DomainError(f"H800 登录依赖不可用：{label}", 503)
        return found

    def principal(self, session_id: str | None) -> Principal | None:
        session = self._lookup(session_id)
        return session.principal() if session else None

    def session(self, session_id: str | None) -> UserSession | None:
        return self._lookup(session_id)

    def session_for_execution(self, execution_id: str | None) -> UserSession | None:
        if not execution_id:
            return None
        with self._lock:
            self._cleanup_locked()
            return self._by_execution.get(execution_id)

    def logout(self, session_id: str | None) -> None:
        if not session_id:
            return
        with self._lock:
            session = self._by_token.pop(session_id, None)
            if session:
                self._by_execution.pop(session.execution_id, None)
        if session:
            self._exit_control(
                session.control_path,
                session.username,
                session.connect_host,
            )

    def close_all(self) -> None:
        with self._lock:
            sessions = list(
                {row.session_id: row for row in self._by_token.values()}.values()
            )
            self._by_token.clear()
            self._by_execution.clear()
        for session in sessions:
            self._exit_control(
                session.control_path,
                session.username,
                session.connect_host,
            )

    def _lookup(self, session_id: str | None) -> UserSession | None:
        if not session_id:
            return None
        with self._lock:
            self._cleanup_locked()
            return self._by_token.get(session_id)

    def _cleanup_locked(self):
        current = time.time()
        expired = [row for row in self._by_token.values() if row.expires_at <= current]
        for session in expired:
            self._by_token.pop(session.session_id, None)
            self._by_execution.pop(session.execution_id, None)
        for session in expired:
            self._exit_control(
                session.control_path,
                session.username,
                session.connect_host,
            )

    def command_argv(
        self,
        session: UserSession,
        argv,
        *,
        cwd=None,
        remote_env: dict | None = None,
    ) -> list[str]:
        active = self.session(session.session_id)
        if active is None or active.execution_id != session.execution_id:
            raise DomainError("H800 用户会话已失效，请重新登录", 401)
        arguments = [str(value) for value in argv]
        command = shlex.join(arguments)
        if remote_env:
            assignments = []
            for key, value in remote_env.items():
                if not _ENVIRONMENT_NAME.fullmatch(str(key)):
                    raise ValueError(f"无效的远端环境变量名：{key}")
                assignments.append(f"{key}={shlex.quote(str(value))}")
            command = "env " + " ".join(assignments) + " " + command
        if cwd is not None:
            command = f"umask 000; cd {shlex.quote(str(cwd))} && exec {command}"
        else:
            command = f"umask 000; exec {command}"
        remote = "bash -lc " + shlex.quote(command)
        return [
            self.ssh,
            "-S",
            str(session.control_path),
            "-o",
            "ControlMaster=no",
            "-o",
            "BatchMode=yes",
            "-o",
            f"StrictHostKeyChecking={self.strict_host_key_checking}",
            "-p",
            str(self.port),
            "-l",
            session.username,
            session.connect_host or self.connect_host,
            remote,
        ]

    def popen(
        self,
        session: UserSession,
        argv,
        *,
        cwd=None,
        env=None,
        remote_env: dict | None = None,
        **kwargs,
    ) -> subprocess.Popen:
        return subprocess.Popen(
            self.command_argv(session, argv, cwd=cwd, remote_env=remote_env),
            cwd=cwd,
            env=env,
            **kwargs,
        )

    def run(
        self,
        session: UserSession,
        argv,
        *,
        cwd=None,
        env=None,
        remote_env: dict | None = None,
        **kwargs,
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            self.command_argv(session, argv, cwd=cwd, remote_env=remote_env),
            cwd=cwd,
            env=env,
            **kwargs,
        )

    def _run_control(
        self,
        control_path: Path,
        username: str,
        connect_host: str,
        command: str,
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                self.ssh,
                "-S",
                str(control_path),
                "-o",
                "ControlMaster=no",
                "-o",
                "BatchMode=yes",
                "-p",
                str(self.port),
                "-l",
                username,
                connect_host,
                command,
            ],
            capture_output=True,
            text=True,
            timeout=self.connect_timeout + 10,
        )

    def _exit_control(
        self,
        control_path: Path,
        username: str,
        connect_host: str | None = None,
    ) -> None:
        try:
            subprocess.run(
                [
                    self.ssh,
                    "-S",
                    str(control_path),
                    "-O",
                    "exit",
                    "-p",
                    str(self.port),
                    "-l",
                    username,
                    connect_host or self.connect_host,
                ],
                capture_output=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            pass
        control_path.unlink(missing_ok=True)
