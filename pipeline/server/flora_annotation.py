"""Flora-TempSeg + Flora-Captioner execution and result adaptation.

Flora remains a separate Python 3.12 toolchain. The Server prepares LeRobot
media, submits one Episode to Slurm, supervises the process and reads the
timestamped caption JSON. Flora modules are never imported into the API
environment.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from pathlib import Path

from pipeline.shared.core import DomainError

ENGINE_ID = "flora"
MODEL_LABEL = "Flora-TempSeg + Flora-Captioner"
DEFAULT_STAGE = "准备 Flora 标注"
MAX_EPISODES = 100000
_EPISODE_ID = re.compile(r"^episode_(\d{1,6})$")


def _config(settings) -> dict:
    value = getattr(settings, "flora", None)
    return value if isinstance(value, dict) else {}


def source_path(settings) -> Path:
    return Path(str(_config(settings).get("src") or "")).expanduser()


def infer_script(settings) -> Path:
    source = source_path(settings)
    for relative in ("captioner/infer.py", "infer.py"):
        path = source / relative
        if path.is_file():
            return path
    return source / "captioner/infer.py"


def _model_directory_ready(path: Path) -> bool:
    has_config = (path / "config.json").is_file() or (
        path / "adapter_config.json"
    ).is_file()
    files = list(path.glob("*.safetensors"))
    if not has_config or not files or any(not file.is_file() or file.stat().st_size <= 0 for file in files):
        return False
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text())
            weight_files = {
                path / str(value)
                for value in (index.get("weight_map") or {}).values()
            }
            if not weight_files or any(
                not file.is_file() or file.stat().st_size <= 0
                for file in weight_files
            ):
                return False
            total_size = (index.get("metadata") or {}).get("total_size")
            if type(total_size) is int and sum(
                file.stat().st_size for file in weight_files
            ) != total_size:
                return False
        except (OSError, TypeError, ValueError):
            return False
    return True


def _base_model_ready(settings, model_cache: Path) -> bool:
    if _model_directory_ready(model_cache / "tempseg_base"):
        return True
    hf_home = Path(str(_config(settings).get("hf_home") or "")).expanduser()
    snapshots = (
        hf_home
        / "hub"
        / "models--Qwen--Qwen3.5-9B"
        / "snapshots"
    )
    return snapshots.is_dir() and any(_model_directory_ready(path) for path in snapshots.iterdir())


def available(settings) -> tuple[bool, str]:
    config = _config(settings)
    if not config.get("enabled"):
        return False, "未启用 Flora 本地标注"
    source = source_path(settings)
    if not infer_script(settings).is_file():
        return False, f"Flora 源码不存在或不完整：{source}"
    python = Path(str(config.get("python") or "")).expanduser()
    if not python.is_file() or not os.access(python, os.X_OK):
        return False, "未找到 Flora 隔离环境的 Python 解释器"
    for key, default, label in (
        ("srun", "srun", "Slurm srun"),
        ("ffmpeg", "ffmpeg", "ffmpeg"),
        ("ffprobe", "ffprobe", "ffprobe"),
    ):
        command = str(config.get(key) or default)
        found = shutil.which(command)
        if not found and not (Path(command).is_file() and os.access(command, os.X_OK)):
            return False, f"未找到 {label}：{command}"
    model_cache = Path(str(config.get("model_cache") or "")).expanduser()
    for kind in ("tempseg", "captioner"):
        directory = model_cache / kind
        if not _model_directory_ready(directory):
            return False, f"Flora 模型缓存不完整：{directory}"
    if not _base_model_ready(settings, model_cache):
        return False, "Flora 基础模型缓存不完整：tempseg_base / Qwen3.5-9B"
    return True, ""


def provider_spec(settings, annotation: dict) -> dict:
    provider = str(annotation.get("provider") or "").strip()
    profiles = _config(settings).get("profiles") or {}
    spec = profiles.get(provider)
    if not isinstance(spec, dict):
        raise DomainError(
            f"Provider {provider or '（未声明）'} 尚未绑定 Flora 输入 Profile",
            422,
        )
    return spec


def profile_for(settings, annotation: dict) -> dict:
    spec = provider_spec(settings, annotation)
    return {"profile": spec["profile"], "yam32": bool(spec.get("yam32", False))}


def command(
    settings,
    annotation: dict,
    media: dict,
    work: Path,
) -> list[str]:
    ok, why = available(settings)
    if not ok:
        raise DomainError(f"Flora 自动标注不可用：{why}", 503)
    config = _config(settings)
    profile = profile_for(settings, annotation)
    main = Path(str(media.get("main") or ""))
    if not main.is_file():
        raise DomainError("Flora Episode 缺少主相机视频", 422)
    python = str(config["python"])
    infer = infer_script(settings)
    output = Path(work) / f"{media['id']}.result.json"
    argv = [
        str(config.get("srun") or "srun"),
        "--partition",
        str(config.get("partition") or "dev"),
        f"--gres=gpu:{int(config.get('gpus') or 1)}",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task",
        str(int(config.get("cpus_per_task") or 8)),
        python,
        str(infer),
        "--video",
        str(main),
        "--profile",
        profile["profile"],
        "--output",
        str(output),
    ]
    for key in ("left_wrist", "right_wrist"):
        value = str(media.get(key) or "").strip()
        if value:
            argv += [f"--{key.replace('_', '-')}", value]
    if profile["yam32"]:
        argv.append("--yam32")
    instruction = str(annotation.get("note") or "").strip()
    if instruction:
        argv += ["--instruction", instruction]
    return argv


def environment(settings) -> dict:
    config = _config(settings)
    env = {
        key: value
        for key, value in settings.tool_environment().items()
        if not key.startswith(("OPENAI_", "OPENROUTER_", "HF_TOKEN"))
        and key not in {"HUGGING_FACE_HUB_TOKEN", "CUDA_VISIBLE_DEVICES"}
    }
    env.setdefault("PATH", os.defpath)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["FLORA_MODEL_CACHE"] = str(
        Path(str(config.get("model_cache") or "")).expanduser()
    )
    env["HF_HOME"] = str(Path(str(config.get("hf_home") or "")).expanduser())
    if config.get("hf_endpoint"):
        env["HF_ENDPOINT"] = str(config["hf_endpoint"])
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["PYTHONPATH"] = str(source_path(settings))
    env.setdefault("MALLOC_ARENA_MAX", "2")
    return env


def _episode_index(episode_id: str) -> int:
    match = _EPISODE_ID.fullmatch(str(episode_id))
    if match is None:
        raise DomainError("Flora Episode ID 无效", 422)
    return int(match.group(1))


def normalize_result(payload: dict, episode_id: str) -> dict:
    if not isinstance(payload, dict) or not isinstance(payload.get("segments"), list):
        raise DomainError("Flora 返回结果缺少 segments", 502)
    segments = []
    incomplete = 0
    for raw in payload["segments"]:
        if not isinstance(raw, dict):
            raise DomainError("Flora segments 包含无效条目", 502)
        try:
            start = float(raw["start_s"])
            end = float(raw["end_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DomainError("Flora segments 的开始或结束时间无效", 502) from exc
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end <= start
        ):
            raise DomainError("Flora segments 的时间区间无效", 502)
        caption = str(raw.get("caption") or "").strip()
        status = str(raw.get("status") or "").strip() or "incomplete"
        if status != "complete":
            incomplete += 1
        segments.append(
            {
                "start_s": start,
                "end_s": end,
                "caption": caption,
                "status": status,
            }
        )
    duration = payload.get("duration_s")
    try:
        duration = float(duration)
    except (TypeError, ValueError):
        duration = None
    revisions = payload.get("model_revisions")
    if not isinstance(revisions, dict):
        revisions = {}
    complete = sum(segment["status"] == "complete" for segment in segments)
    return {
        "id": episode_id,
        "episode_id": episode_id,
        "episode_index": _episode_index(episode_id),
        "duration_s": duration,
        "labels_available": complete > 0,
        "completed": complete > 0 and incomplete == 0,
        "segments": segments,
        "model": MODEL_LABEL,
        "profile": payload.get("profile"),
        "model_revisions": revisions,
        "source_sha256": payload.get("input_sha256"),
        "label_failure": (
            {"status": "partial", "incomplete_segments": incomplete}
            if incomplete
            else None
        ),
    }


def outputs(work: Path) -> list[dict]:
    rows = []
    for path in sorted(Path(work).glob("episode_*.result.json"))[:MAX_EPISODES]:
        episode_id = path.name.removesuffix(".result.json")
        try:
            payload = json.loads(path.read_text())
            rows.append(normalize_result(payload, episode_id))
        except (OSError, ValueError, DomainError):
            continue
    return rows


def results(work: Path) -> dict[str, dict]:
    return {row["id"]: row for row in outputs(work)}


def episode(work: Path, episode_id: str) -> dict | None:
    _episode_index(episode_id)
    path = Path(work) / f"{episode_id}.result.json"
    try:
        return normalize_result(json.loads(path.read_text()), episode_id)
    except (OSError, ValueError, DomainError):
        return None


def progress(work: Path) -> dict:
    completed = 0
    count = 0
    for path in Path(work).glob("episode_*.result.json"):
        count += 1
        try:
            payload = json.loads(path.read_text())
            row = normalize_result(payload, path.name.removesuffix(".result.json"))
        except (OSError, ValueError, DomainError):
            continue
        completed += int(row["labels_available"])
    return {"units": count, "labelled": completed, "episodes": completed}


def clips(_work: Path, _episode_id: str) -> list[dict]:
    return []


def clip(_work: Path, _episode_id: str, _camera: str) -> Path | None:
    return None


def log_names(work: Path) -> list[str]:
    logs = Path(work) / "flora/logs"
    names = ["run"]
    if logs.is_dir():
        names.extend(sorted(path.stem for path in logs.glob("*.log")))
    return names


def read_log(work: Path, name: str) -> str:
    if name not in log_names(work):
        raise DomainError("日志不存在", 404)
    path = (
        Path(work) / "logs/run.log"
        if name == "run"
        else Path(work) / "flora/logs" / f"{name}.log"
    )
    try:
        return path.read_text(errors="replace")[-64000:]
    except OSError:
        return ""


def stage_from_log(text: str, current: str = "") -> str:
    stage = current or DEFAULT_STAGE
    if "Preparing LeRobot media" in text:
        stage = "准备 Episode 视频"
    if "FLORA_RUNTIME" in text or "Loading weights" in text:
        stage = "Flora 推理"
    if "Captioning" in text or "captioner" in text.lower():
        stage = "生成自然语言字幕"
    if '"segments"' in text:
        stage = "拼接标注结果"
    return stage


def cost(_work: Path) -> dict:
    return {
        "status": "done",
        "cost_usd": 0.0,
        "reserved_cost_usd": 0.0,
        "cost_complete": True,
    }


__all__ = [
    "ENGINE_ID",
    "MODEL_LABEL",
    "available",
    "clip",
    "clips",
    "command",
    "cost",
    "environment",
    "episode",
    "infer_script",
    "log_names",
    "normalize_result",
    "outputs",
    "profile_for",
    "provider_spec",
    "progress",
    "read_log",
    "results",
    "source_path",
    "stage_from_log",
]
