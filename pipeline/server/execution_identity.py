"""Runtime gate for attributing new Viewer compute to browser users."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import Request

    from pipeline.shared.config import Settings
    from pipeline.shared.identity import Principal


def viewer_identity_enabled(settings: Settings) -> bool:
    config = getattr(settings, "viewer_execution_identity", {})
    return isinstance(config, dict) and config.get("enabled") is True


def execution_principal(
    settings: Settings, request: Request
) -> Principal | None:
    principal = getattr(request.state, "principal", None)
    return principal if viewer_identity_enabled(settings) else None
