"""Discover LeRobot Episodes and prepare synchronized Flora input videos."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pipeline.shared.core import DomainError

MAX_DURATION_S = 120.0
MAX_FRAMES = 7200
MAX_FILE_BYTES = 256 * 1024 * 1024
MAX_SIDE = 4096
MIN_FPS = 4.0
CAMERA_ROLES = ("main", "left_wrist", "right_wrist")


@dataclass(frozen=True)
class CameraClip:
    path: Path
    start_s: float
    end_s: float


@dataclass(frozen=True)
class EpisodeMedia:
    episode_id: str
    episode_index: int
    duration_s: float
    cameras: dict[str, CameraClip]


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise DomainError(f"无法读取 {path}：{exc}", 422) from exc
    if not isinstance(value, dict):
        raise DomainError(f"{path} 必须包含 JSON 对象", 422)
    return value


def _video_keys(info: dict) -> list[str]:
    return sorted(
        key
        for key, spec in (info.get("features") or {}).items()
        if isinstance(spec, dict) and spec.get("dtype") == "video"
    )


def _infer_camera_roles(keys: list[str]) -> dict[str, str]:
    def pick(*needles):
        for key in keys:
            low = key.lower()
            if all(needle in low for needle in needles):
                return key
        return None

    main = (
        pick("front")
        or pick("head")
        or pick("scene")
        or pick("top")
        or pick("cam_high")
        or next(
            (
                key
                for key in keys
                if "wrist" not in key.lower() and "left" not in key.lower()
                and "right" not in key.lower()
            ),
            None,
        )
    )
    left = pick("left", "wrist") or pick("wrist", "left")
    right = pick("right", "wrist") or pick("wrist", "right")
    return {
        role: key
        for role, key in (
            ("main", main),
            ("left_wrist", left),
            ("right_wrist", right),
        )
        if key
    }


def _resolve_camera_roles(info: dict, camera_roles: dict | None) -> dict[str, str]:
    keys = _video_keys(info)
    if not keys:
        raise DomainError("数据集没有可标注的视频相机", 422)
    roles = dict(camera_roles or _infer_camera_roles(keys))
    unknown_roles = set(roles) - set(CAMERA_ROLES)
    if unknown_roles:
        raise DomainError(
            f"Flora camera_roles 包含未知角色：{', '.join(sorted(unknown_roles))}",
            422,
        )
    unknown_cameras = {key for key in roles.values() if key not in keys}
    if unknown_cameras:
        raise DomainError(
            f"Flora camera_roles 指向不存在的相机：{', '.join(sorted(unknown_cameras))}",
            422,
        )
    if not roles.get("main"):
        raise DomainError("Flora 至少需要一个主相机", 422)
    return roles


def _v21_episode_rows(root: Path) -> list[dict]:
    path = root / "meta/episodes.jsonl"
    try:
        rows = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]
    except (OSError, ValueError) as exc:
        raise DomainError(f"无法读取 LeRobot v2.1 Episode 元数据：{exc}", 422) from exc
    return [row for row in rows if isinstance(row, dict)]


def _parquet_rows(root: Path) -> list[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise DomainError(
            "读取 LeRobot v3.0 Episode 元数据需要 pyarrow；"
            "请使用安装脚本中的隔离环境运行 Flora",
            503,
        ) from exc
    rows = []
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def load_episode_manifest(
    dataset: Path,
    *,
    camera_roles: dict | None = None,
) -> list[EpisodeMedia]:
    root = Path(dataset)
    info = _read_json(root / "meta/info.json")
    version = info.get("codebase_version")
    if version not in {"v2.1", "v3.0"}:
        raise DomainError("Flora 仅支持 LeRobot v2.1 / v3.0", 422)
    fps = float(info.get("fps") or 0)
    if fps <= 0:
        raise DomainError("LeRobot fps 无效", 422)
    roles = _resolve_camera_roles(info, camera_roles)
    rows = _v21_episode_rows(root) if version == "v2.1" else _parquet_rows(root)
    chunks_size = max(1, int(info.get("chunks_size") or 1000))
    episodes = []
    for row in rows:
        try:
            episode_index = int(row["episode_index"])
            length = int(row["length"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DomainError("LeRobot Episode 缺少有效 episode_index/length", 422) from exc
        if episode_index < 0 or length <= 0:
            raise DomainError("LeRobot Episode 的索引或长度无效", 422)
        cameras = {}
        if version == "v2.1":
            episode_chunk = episode_index // chunks_size
            template = str(
                info.get("video_path")
                or "videos/{video_key}/chunk-{episode_chunk:03d}/"
                "episode_{episode_index:06d}.mp4"
            )
            for role, camera in roles.items():
                path = root / template.format(
                    video_key=camera,
                    episode_chunk=episode_chunk,
                    episode_index=episode_index,
                )
                cameras[role] = CameraClip(
                    path=path,
                    start_s=0.0,
                    end_s=length / fps,
                )
        else:
            template = str(
                info.get("video_path")
                or "videos/{video_key}/chunk-{chunk_index:03d}/"
                "file-{file_index:03d}.mp4"
            )
            for role, camera in roles.items():
                prefix = f"videos/{camera}"
                try:
                    chunk = int(row[f"{prefix}/chunk_index"])
                    file_index = int(row[f"{prefix}/file_index"])
                    start = float(row[f"{prefix}/from_timestamp"])
                    end = float(row[f"{prefix}/to_timestamp"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise DomainError(
                        f"Episode {episode_index} 缺少 {camera} 视频区间", 422
                    ) from exc
                path = root / template.format(
                    video_key=camera,
                    chunk_index=chunk,
                    file_index=file_index,
                    episode_index=episode_index,
                )
                cameras[role] = CameraClip(path=path, start_s=start, end_s=end)
        for role, clip in cameras.items():
            if not clip.path.is_file():
                raise DomainError(
                    f"Episode {episode_index} 的 {role} 视频不存在：{clip.path}",
                    422,
                )
            if clip.start_s < 0 or clip.end_s <= clip.start_s:
                raise DomainError(
                    f"Episode {episode_index} 的 {role} 时间区间无效", 422
                )
        duration = cameras["main"].end_s - cameras["main"].start_s
        episodes.append(
            EpisodeMedia(
                episode_id=f"episode_{episode_index:06d}",
                episode_index=episode_index,
                duration_s=round(duration, 6),
                cameras=cameras,
            )
        )
    return episodes


def probe_video(path: Path, ffprobe: str = "ffprobe") -> dict:
    argv = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,nb_frames,duration",
        "-show_entries",
        "format=duration,size",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            argv,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise DomainError(f"无法探测 Flora 输入视频 {path}：{exc}", 422) from exc
    streams = payload.get("streams") if isinstance(payload, dict) else None
    if not isinstance(streams, list) or not streams:
        raise DomainError(f"视频没有视频流：{path}", 422)
    stream = streams[0]
    fmt = payload.get("format") if isinstance(payload.get("format"), dict) else {}
    rate = str(stream.get("avg_frame_rate") or "0/1")
    try:
        numerator, denominator = (int(part) for part in rate.split("/", 1))
        fps = numerator / denominator if denominator else 0.0
    except (TypeError, ValueError):
        fps = 0.0
    duration = stream.get("duration") or fmt.get("duration")
    try:
        duration = float(duration)
    except (TypeError, ValueError):
        duration = 0.0
    try:
        frames = int(stream.get("nb_frames") or round(duration * fps))
    except (TypeError, ValueError):
        frames = 0
    return {
        "duration_s": duration,
        "fps": fps,
        "frames": frames,
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "bytes": int(fmt.get("size") or path.stat().st_size),
        "codec": stream.get("codec_name"),
    }


def _validate_probe(path: Path, probe: dict):
    if (
        float(probe.get("duration_s") or 0) <= 0
        or float(probe.get("fps") or 0) < MIN_FPS
        or int(probe.get("frames") or 0) <= 0
    ):
        raise DomainError(f"Flora 输入视频参数无效：{path}", 422)
    if max(int(probe.get("width") or 0), int(probe.get("height") or 0)) > MAX_SIDE:
        raise DomainError(f"Flora 输入视频边长超过 4096 px：{path}", 422)


def _clip_argv(
    *,
    ffmpeg: str,
    source: Path,
    output: Path,
    start_s: float,
    duration_s: float,
    codec: str,
) -> list[str]:
    common = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(source),
        "-t",
        f"{duration_s:.3f}",
        "-map",
        "0:v:0",
        "-an",
    ]
    if codec == "copy":
        return [*common, "-c", "copy", "-y", str(output)]
    return [
        *common,
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-y",
        str(output),
    ]


def _extract_clip(argv: list[str], output: Path, run: Callable[[list[str]], None]):
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        run(argv)
    except (OSError, subprocess.CalledProcessError):
        output.unlink(missing_ok=True)
        retry = list(argv)
        if "-c" not in retry:
            raise
        codec_at = retry.index("-c")
        retry[codec_at : codec_at + 2] = [
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
        ]
        run(retry)
    if not output.is_file():
        raise DomainError(f"FFmpeg 未生成 Flora 视频：{output}", 500)
    size = output.stat().st_size
    if size <= 0 or size > MAX_FILE_BYTES:
        output.unlink(missing_ok=True)
        raise DomainError(f"Flora 输入视频超过 256 MiB 或为空：{output}", 422)


def prepare_episode_slices(
    episode: EpisodeMedia,
    destination: Path,
    *,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    probe: Callable[[Path], dict] | None = None,
    run: Callable[[list[str]], None] | None = None,
) -> list[dict]:
    probe = probe or (lambda path: probe_video(path, ffprobe))
    run = run or (
        lambda argv: subprocess.run(argv, check=True, stdout=subprocess.DEVNULL)
    )
    probes = {}
    for role, clip in episode.cameras.items():
        value = probe(clip.path)
        _validate_probe(clip.path, value)
        clip_duration = clip.end_s - clip.start_s
        if abs(clip_duration - episode.duration_s) > max(
            0.05, 1 / float(value["fps"])
        ):
            raise DomainError(
                f"Episode {episode.episode_index} 相机 {role} 与 Episode 时长不同步",
                422,
            )
        if clip.end_s > float(value["duration_s"]) + max(
            0.05, 1 / float(value["fps"])
        ):
            raise DomainError(
                f"Episode {episode.episode_index} 的 {role} 视频区间超出源文件",
                422,
            )
        probes[role] = value
    first_path, first = next(iter(probes.items()))
    for role, value in probes.items():
        fps_delta = abs(float(value["fps"]) - float(first["fps"]))
        if fps_delta > 0.01:
            raise DomainError(
                f"Episode {episode.episode_index} 相机不同步："
                f"{first_path} 与 {role} 的帧率不一致",
                422,
            )
    source_duration = float(episode.duration_s)
    source_frames = round(source_duration * float(first["fps"]))
    if (
        source_duration > MAX_DURATION_S + 1e-6
        or source_frames > MAX_FRAMES
    ):
        count = max(
            math.ceil(source_duration / MAX_DURATION_S),
            math.ceil(source_frames / MAX_FRAMES),
        )
    else:
        count = 1
    rows = []
    for number in range(count):
        offset = number * MAX_DURATION_S
        duration = min(MAX_DURATION_S, source_duration - offset)
        if duration <= 0:
            break
        folder = Path(destination) / episode.episode_id / f"slice_{number:03d}"
        row = {
            "id": f"{episode.episode_id}_slice_{number:03d}",
            "episode_id": episode.episode_id,
            "episode_index": episode.episode_index,
            "slice_index": number,
            "offset_s": round(offset, 6),
            "duration_s": round(duration, 6),
        }
        for role, clip in episode.cameras.items():
            output = folder / f"{role}.mp4"
            argv = _clip_argv(
                ffmpeg=ffmpeg,
                source=clip.path,
                output=output,
                start_s=clip.start_s + offset,
                duration_s=duration,
                codec="copy",
            )
            _extract_clip(argv, output, run)
            row[role] = str(output)
        rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--main-camera")
    parser.add_argument("--left-wrist-camera")
    parser.add_argument("--right-wrist-camera")
    parser.add_argument("--episode-index", type=int)
    args = parser.parse_args(argv)
    roles = {
        role: key
        for role, key in (
            ("main", args.main_camera),
            ("left_wrist", args.left_wrist_camera),
            ("right_wrist", args.right_wrist_camera),
        )
        if key
    }
    episodes = load_episode_manifest(
        args.dataset,
        camera_roles=roles or None,
    )
    if args.episode_index is not None:
        episodes = [
            row for row in episodes if row.episode_index == args.episode_index
        ]
    slices = []
    for episode in episodes:
        slices.extend(prepare_episode_slices(episode, args.output))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "media.json").write_text(
        json.dumps({"slices": slices}, indent=2, ensure_ascii=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
