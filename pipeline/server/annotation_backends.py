"""Deployment-controlled annotation backend registry.

The queue owns persistence, cancellation, resume and export. Backends own the
engine-specific command, credentials and sidecar adaptation.
"""

from __future__ import annotations

import functools
import subprocess
import tempfile
import time
from pathlib import Path

from pipeline.shared.core import DomainError, digest, executable_path

from . import argus_annotation, codex_transport, flora_annotation


class AnnotationBackend:
    id: str
    label: str

    def available(self, settings) -> tuple[bool, str]: ...

    def results(self, work) -> dict[str, dict]: ...

    def clips(self, work, episode_id, settings): ...

    def clip(self, work, episode_id, camera, settings): ...


def backend(row):
    name = str(row.get("backend") or row.get("engine") or "argus")
    if name not in {"argus", "codex"}:
        raise DomainError("未知标注后端；不会回退其他模型后端", 422)
    return name


@functools.lru_cache(maxsize=16)
def _cli_status(executable, bucket):
    if not executable:
        return False, "未找到 Codex CLI；请配置 codex_bin", ""
    try:
        with tempfile.TemporaryDirectory(prefix="bodypipeline-codex-status-") as cwd:
            version = subprocess.run(
                [executable, "--version"],
                cwd=cwd,
                env=codex_transport.cli_environment(),
                capture_output=True,
                text=True,
                timeout=10,
            )
            result = subprocess.run(
                [executable, "login", "status"],
                cwd=cwd,
                env=codex_transport.cli_environment(),
                capture_output=True,
                text=True,
                timeout=10,
            )
        # Do not return arbitrary CLI output, which could include credentials.
        import re

        match = re.search(r"codex-cli [0-9.]+", version.stdout)
        if version.returncode:
            return False, "Codex CLI 无法启动", ""
        if result.returncode:
            return (
                False,
                "Codex CLI 未登录；请用运行服务器的用户执行 codex login",
                match[0] if match else "",
            )
        return True, "已复用服务器用户的 CLI 登录", match[0] if match else ""
    except (OSError, subprocess.TimeoutExpired):
        return False, "Codex CLI 状态检查失败或超时", ""


def codex_options(settings):
    executable = executable_path(settings.codex_bin)
    ok, reason, version = _cli_status(executable, int(time.monotonic() // 30))
    try:
        public = codex_transport.public_configuration()
    except codex_transport.CLIError as exc:
        return {"available": False, "reason": str(exc), "models": []}
    model = settings.codex_model or public.get("model") or ""
    reasoning = (
        settings.codex_reasoning or public.get("model_reasoning_effort") or "medium"
    )
    models = list(settings.codex_models or [model])
    if not model or model not in models:
        ok, reason = False, "请在 CLI 或部署配置中指定可用的 Codex 默认模型"
    if not settings.argus_annotation_unlimited:
        ok, reason = False, "Codex CLI 不能核实金额或执行费用上限；需部署启用不限额标注"
    common, why = argus_annotation.available(settings)
    helper = Path(settings.argus_src) / "label/pipeline_codex.py"
    try:
        installed = helper.read_bytes() == Path(codex_transport.__file__).read_bytes()
        installed &= (
            "pipeline-argus-codex:v1"
            in (Path(settings.argus_src) / "label/harness.py").read_text()
        )
        installed &= (
            "codex part request budget"
            in (Path(settings.argus_src) / "label/policy.py").read_text()
        )
    except OSError:
        installed = False
    if not common or not installed:
        ok, reason = False, why or "Argus 缺少 Codex transport；请安装集成补丁 0005"
    return {
        "available": ok,
        "reason": reason,
        "version": version,
        "models": models,
        "default_model": model,
        "default_reasoning": reasoning,
        "provider": public.get("model_provider", "openai"),
        "provider_url": (public.get("provider_config") or {}).get("base_url"),
        "capabilities": {
            "templates": True,
            "free_analysis": False,
            "inspection": ["off"],
            "field_dictionary": False,
            "max_tokens": False,
            "reasoning": ["low", "medium", "high", "xhigh"],
        },
        "defaults": {"concurrency": 1, "sample_fps": 3.0},
        "limits": {
            "max_images": settings.codex_max_images,
            "max_image_bytes": settings.codex_max_image_bytes,
            "max_prompt_chars": settings.codex_max_prompt_chars,
            "timeout": settings.codex_timeout,
            "max_part_seconds": settings.codex_max_part_seconds,
        },
        "notice": "本机指 CLI 在服务器上运行；模型计算可能由 Codex 服务完成。复用 CLI 登录和模型/提供方配置，隔离其工具、项目规则与插件。实际金额未知。",
    }


def codex_snapshot(settings, model, reasoning):
    public = codex_transport.public_configuration()
    return {
        "version": codex_transport.VERSION,
        "executable": executable_path(settings.codex_bin),
        "model": model,
        "reasoning": reasoning,
        "model_provider": public.get("model_provider", "openai"),
        "provider_config": public.get("provider_config", {}),
        "cli_settings": public.get("cli_settings", {}),
        "max_images": settings.codex_max_images,
        "max_image_bytes": settings.codex_max_image_bytes,
        "max_prompt_chars": settings.codex_max_prompt_chars,
        "max_output_bytes": settings.codex_max_output_bytes,
        "timeout": settings.codex_timeout,
        "max_part_seconds": settings.codex_max_part_seconds,
    }


def ensure_snapshot(settings, row):
    if backend(row) != "codex":
        return
    current = codex_snapshot(settings, row["model"], row["reasoning"])
    if digest(current) != digest(row.get("backend_config")):
        raise DomainError(
            "Codex 部署模型、提供方或输入限制已变化；请创建新任务，不会静默更换续跑配置",
            409,
        )


def cost(work):
    rows = codex_transport.receipts(Path(work) / "codex-calls")
    return {
        "cost_usd": None,
        "reserved_cost_usd": None,
        "cost_complete": False,
        "cost_source": "unknown_codex_cli",
        "cli_calls": rows,
        "cli_call_count": len(rows),
        "cli_usage": {
            k: sum((r.get("usage") or {}).get(k, 0) or 0 for r in rows)
            for k in ("input_tokens", "cached_input_tokens", "output_tokens")
        },
    }


class ArgusBackend(AnnotationBackend):
    id = "argus"
    label = "Argus 云端"

    def available(self, settings):
        return argus_annotation.available(settings)

    outputs = staticmethod(argus_annotation.outputs)
    results = staticmethod(argus_annotation.episode_results)
    episode = staticmethod(argus_annotation.episode)

    @staticmethod
    def clips(work, episode_id, _settings=None):
        return argus_annotation.clips(work, episode_id)

    @staticmethod
    def clip(work, episode_id, camera, _settings=None):
        return argus_annotation.clip(work, episode_id, camera)

    log_names = staticmethod(argus_annotation.log_names)
    read_log = staticmethod(argus_annotation.read_log)
    progress = staticmethod(argus_annotation.progress)
    stage_from_log = staticmethod(argus_annotation.stage_from_log)
    cost = staticmethod(argus_annotation.cost)


class CodexBackend(AnnotationBackend):
    id = "codex"
    label = "本机 Codex"

    def available(self, settings):
        status = codex_options(settings)
        return status["available"], status["reason"]

    # The pinned Argus runner remains the transport host. Codex model
    # credentials and provider configuration stay in the server-local CLI.
    outputs = staticmethod(argus_annotation.outputs)
    results = staticmethod(argus_annotation.episode_results)
    episode = staticmethod(argus_annotation.episode)

    @staticmethod
    def clips(work, episode_id, _settings=None):
        return argus_annotation.clips(work, episode_id)

    @staticmethod
    def clip(work, episode_id, camera, _settings=None):
        return argus_annotation.clip(work, episode_id, camera)

    log_names = staticmethod(argus_annotation.log_names)
    read_log = staticmethod(argus_annotation.read_log)
    progress = staticmethod(argus_annotation.progress)
    stage_from_log = staticmethod(argus_annotation.stage_from_log)
    cost = staticmethod(cost)


class FloraBackend(AnnotationBackend):
    id = flora_annotation.ENGINE_ID
    label = "Flora 本地"

    def available(self, settings):
        return flora_annotation.available(settings)

    outputs = staticmethod(flora_annotation.outputs)
    results = staticmethod(flora_annotation.results)
    episode = staticmethod(flora_annotation.episode)

    @staticmethod
    def clips(work, episode_id, settings=None):
        return flora_annotation.clips(work, episode_id, settings)

    @staticmethod
    def clip(work, episode_id, camera, settings=None):
        return flora_annotation.clip(work, episode_id, camera, settings)

    log_names = staticmethod(flora_annotation.log_names)
    read_log = staticmethod(flora_annotation.read_log)
    progress = staticmethod(flora_annotation.progress)
    stage_from_log = staticmethod(flora_annotation.stage_from_log)
    cost = staticmethod(flora_annotation.cost)


BACKENDS = {
    ArgusBackend.id: ArgusBackend(),
    CodexBackend.id: CodexBackend(),
    FloraBackend.id: FloraBackend(),
}
# Local Codex remains readable for legacy records, but it is not exposed as a
# new automatic-annotation engine.
CREATE_BACKENDS = (ArgusBackend.id, FloraBackend.id)
DEFAULT_ENGINE = ArgusBackend.id
DEFAULT_BACKEND = ArgusBackend.id


def backend_for(annotation: dict) -> AnnotationBackend:
    engine = str(
        annotation.get("engine") or annotation.get("backend") or DEFAULT_ENGINE
    )
    result = BACKENDS.get(engine)
    if result is None:
        raise DomainError(f"未知标注引擎：{engine}", 400)
    return result
