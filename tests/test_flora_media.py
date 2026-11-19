"""LeRobot Episode discovery and bounded media preparation for Flora."""

import json
from pathlib import Path

import pytest

from pipeline.server import flora_media_worker
from pipeline.shared.core import DomainError


def make_v21_fixture(root: Path, *, frames=3630):
    (root / "meta").mkdir(parents=True)
    features = {
        "observation.images.front": {"dtype": "video"},
        "observation.images.left_wrist": {"dtype": "video"},
        "observation.images.right_wrist": {"dtype": "video"},
        "action": {"dtype": "float32", "shape": [14]},
    }
    fps = 30
    info = {
        "codebase_version": "v2.1",
        "robot_type": "fixture",
        "total_episodes": 1,
        "total_frames": frames,
        "fps": fps,
        "chunks_size": 1000,
        "features": features,
        "video_path": (
            "videos/{video_key}/chunk-{episode_chunk:03d}/"
            "episode_{episode_index:06d}.mp4"
        ),
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": frames, "tasks": ["Put."]})
        + "\n"
    )
    for camera in features:
        if not camera.startswith("observation.images."):
            continue
        path = root / f"videos/{camera}/chunk-000/episode_000000.mp4"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"video")
    return info


def test_v21_manifest_maps_semantic_cameras_and_episode_time(tmp_path):
    root = tmp_path / "dataset"
    make_v21_fixture(root)
    rows = flora_media_worker.load_episode_manifest(
        root,
        camera_roles={
            "main": "observation.images.front",
            "left_wrist": "observation.images.left_wrist",
            "right_wrist": "observation.images.right_wrist",
        },
    )
    assert len(rows) == 1
    episode = rows[0]
    assert episode.episode_id == "episode_000000"
    assert episode.duration_s == 121
    assert episode.cameras["main"].start_s == 0
    assert episode.cameras["main"].end_s == 121
    assert episode.cameras["main"].path == (
        root
        / "videos/observation.images.front/chunk-000/episode_000000.mp4"
    )


def test_v30_manifest_reads_per_camera_shared_file_ranges(
    tmp_path, monkeypatch
):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "fps": 30,
        "total_episodes": 1,
        "total_frames": 60,
        "chunks_size": 1000,
        "features": {
            "observation.images.front": {"dtype": "video"},
            "observation.images.left_wrist": {"dtype": "video"},
        },
        "video_path": (
            "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        ),
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    for camera in ("front", "left_wrist"):
        path = root / f"videos/observation.images.{camera}/chunk-000/file-000.mp4"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"video")
    monkeypatch.setattr(
        flora_media_worker,
        "_parquet_rows",
        lambda _root: [
            {
                "episode_index": 3,
                "length": 60,
                "videos/observation.images.front/chunk_index": 0,
                "videos/observation.images.front/file_index": 0,
                "videos/observation.images.front/from_timestamp": 5.0,
                "videos/observation.images.front/to_timestamp": 7.0,
                "videos/observation.images.left_wrist/chunk_index": 0,
                "videos/observation.images.left_wrist/file_index": 0,
                "videos/observation.images.left_wrist/from_timestamp": 5.0,
                "videos/observation.images.left_wrist/to_timestamp": 7.0,
            }
        ],
    )

    rows = flora_media_worker.load_episode_manifest(
        root,
        camera_roles={
            "main": "observation.images.front",
            "left_wrist": "observation.images.left_wrist",
        },
    )
    assert rows[0].episode_id == "episode_000003"
    assert rows[0].duration_s == 2
    assert rows[0].cameras["left_wrist"].start_s == 5
    assert rows[0].cameras["left_wrist"].end_s == 7


def test_camera_duration_mismatch_is_rejected(tmp_path):
    root = tmp_path / "dataset"
    make_v21_fixture(root, frames=60)
    original = flora_media_worker.load_episode_manifest(root)
    assert original[0].duration_s == 2

    def mismatch(path):
        if "front" in str(path):
            return {"duration_s": 2, "fps": 30, "frames": 60}
        return {"duration_s": 2.1, "fps": 30, "frames": 63}

    with pytest.raises(DomainError, match="同步"):
        flora_media_worker.prepare_episode_slices(
            original[0],
            root / "prepared",
            probe=mismatch,
            run=lambda _argv: None,
        )


def test_long_episode_is_split_into_bounded_slices_with_absolute_offsets(tmp_path):
    root = tmp_path / "dataset"
    make_v21_fixture(root)
    episode = flora_media_worker.load_episode_manifest(root)[0]
    calls = []

    def run(argv):
        calls.append(list(argv))
        Path(argv[-1]).parent.mkdir(parents=True, exist_ok=True)
        Path(argv[-1]).write_bytes(b"clip")

    rows = flora_media_worker.prepare_episode_slices(
        episode,
        root / "prepared",
        probe=lambda _path: {"duration_s": 121, "fps": 30, "frames": 3630},
        run=run,
    )
    assert [row["offset_s"] for row in rows] == [0.0, 120.0]
    assert [row["duration_s"] for row in rows] == [120.0, 1.0]
    assert rows[1]["main"].endswith("slice_001/main.mp4")
    assert all("-c" in call and call[call.index("-c") + 1] == "copy" for call in calls)
    second_slice = next(
        call for call in calls if any("slice_001/main.mp4" in arg for arg in call)
    )
    assert second_slice[second_slice.index("-ss") + 1] == "120.000"
    assert second_slice[second_slice.index("-t") + 1] == "1.000"


def test_copy_failure_retries_with_an_explicit_reencode(tmp_path):
    root = tmp_path / "dataset"
    make_v21_fixture(root, frames=60)
    episode = flora_media_worker.load_episode_manifest(root)[0]
    calls = []

    def run(argv):
        calls.append(list(argv))
        if "copy" in argv:
            raise OSError("copy failed")
        Path(argv[-1]).parent.mkdir(parents=True, exist_ok=True)
        Path(argv[-1]).write_bytes(b"clip")

    rows = flora_media_worker.prepare_episode_slices(
        episode,
        root / "prepared",
        probe=lambda _path: {"duration_s": 2, "fps": 30, "frames": 60},
        run=run,
    )
    assert len(rows) == 1
    assert any("libx264" in call for call in calls)
    assert any("-crf" in call for call in calls)
