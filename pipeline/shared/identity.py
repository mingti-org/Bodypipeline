from __future__ import annotations

from dataclasses import dataclass

from .core import DomainError


@dataclass(frozen=True)
class Principal:
    """Authenticated caller used by the Server request and job boundaries.

    ``execution_id`` is an internal handle for a private SSH control socket. It
    is deliberately separate from the opaque browser cookie so durable job
    records never contain a reusable login token.
    """

    username: str
    uid: int | None = None
    login_node: str | None = None
    execution_id: str | None = None
    auth_method: str = "service-token"
    is_admin: bool = False

    def public(self) -> dict:
        return {
            "username": self.username,
            "uid": self.uid,
            "login_node": self.login_node,
            "auth_method": self.auth_method,
            "is_admin": self.is_admin,
        }

    def job_metadata(self) -> dict:
        if self.is_admin:
            return {
                "owner_user_id": None,
                "owner_uid": None,
                "submitted_as": "service-account",
                "slurm_user": None,
                "slurm_execution_id": None,
                "submission_auth": self.auth_method,
            }
        return {
            "owner_user_id": self.username,
            "owner_uid": self.uid,
            "submitted_as": self.username,
            "slurm_user": self.username,
            "slurm_execution_id": self.execution_id,
            "submission_auth": self.auth_method,
        }


def require_owner(row: dict, principal: Principal, *, action: str = "操作") -> None:
    """Reject a non-admin caller who did not submit this task.

    Records created before identity support have no owner and remain readable,
    but mutating them requires the service token so a shared login cannot grant
    access to historical administrative actions.
    """

    if principal.is_admin:
        return
    owner = row.get("owner_user_id")
    if owner and owner == principal.username:
        return
    raise DomainError(f"只有任务发起者或管理员可以{action}此任务", 403)
