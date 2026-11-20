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


def test_v30_manifest_reads_real_episode_parquet(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "dataset"
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
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
    row = {
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
    pq.write_table(pa.Table.from_pylist([row]), root / "meta/episodes/chunk-000/file-000.parquet")

    rows = flora_media_worker.load_episode_manifest(
        root,
        camera_roles={
            "main": "observation.images.front",
            "left_wrist": "observation.images.left_wrist",
        },
    )
    assert rows[0].episode_id == "episode_000003"
    assert rows[0].duration_s == 2
    assert rows[0].cameras["main"].start_s == 5
    assert rows[0].cameras["left_wrist"].end_s == 7


def test_camera_duration_mismatch_is_rejected(tmp_path):
    root = tmp_path / "dataset"
    make_v21_fixture(root, frames=60)
    original = flora_media_worker.load_episode_manifest(root)
    assert original[0].duration_s == 2

    def mismatch(path):
        if "front" in str(path):
            return {"duration_s": 2, "fps": 30, "frames": 60}
        return {"duration_s": 2, "fps": 29, "frames": 58}

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


def test_v30_shared_video_is_sliced_by_episode_range_not_file_duration(tmp_path):
    root = tmp_path / "dataset"
    video = root / "videos/front/chunk-000/file-000.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"shared-video")
    episode = flora_media_worker.EpisodeMedia(
        episode_id="episode_000004",
        episode_index=4,
        duration_s=2,
        cameras={
            "main": flora_media_worker.CameraClip(video, 120, 122),
        },
    )
    calls = []

    def run(argv):
        calls.append(list(argv))
        Path(argv[-1]).parent.mkdir(parents=True, exist_ok=True)
        Path(argv[-1]).write_bytes(b"clip")

    rows = flora_media_worker.prepare_episode_slices(
        episode,
        root / "prepared",
        probe=lambda _path: {
            "duration_s": 600,
            "fps": 30,
            "frames": 18000,
            "width": 640,
            "height": 480,
        },
        run=run,
    )
    assert len(rows) == 1
    assert rows[0]["duration_s"] == 2
    assert calls[0][calls[0].index("-ss") + 1] == "120.000"
    assert calls[0][calls[0].index("-t") + 1] == "2.000"


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


def test_episode_previews_concatenate_all_slices_for_each_camera(tmp_path):
    root = tmp_path / "dataset"
    make_v21_fixture(root, frames=120)
    episode = flora_media_worker.load_episode_manifest(root)[0]
    slices = []
    for index in range(2):
        folder = tmp_path / f"media/episode_000000/slice_{index:03d}"
        folder.mkdir(parents=True)
        row = {"offset_s": index * 2.0}
        for role in ("main", "left_wrist", "right_wrist"):
            path = folder / f"{role}.mp4"
            path.write_bytes(f"{role}-{index}".encode())
            row[role] = str(path)
        slices.append(row)

    calls = []
    concat_contents = []

    def run(argv):
        calls.append(list(argv))
        concat_file = Path(argv[argv.index("-i") + 1])
        concat_contents.append(concat_file.read_text())
        Path(argv[-1]).write_bytes(b"combined")

    previews = flora_media_worker.prepare_episode_previews(
        episode,
        slices,
        tmp_path / "previews",
        ffmpeg="ffmpeg",
        ffprobe="ffprobe",
        probe=lambda _path: {"duration_s": 4, "fps": 30, "frames": 120},
        run=run,
    )

    assert list(previews) == ["main", "left_wrist", "right_wrist"]
    assert all(path.read_bytes() == b"combined" for path in previews.values())
    assert len(calls) == 3
    assert all(contents.count("file '") == 2 for contents in concat_contents)
