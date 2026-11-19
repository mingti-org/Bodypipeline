"""Flora annotation backend contracts and output adaptation."""

import json

import pytest

from pipeline.server import flora_annotation
from pipeline.shared.core import DomainError


def make_flora(tmp_path):
    source = tmp_path / "flora"
    (source / "captioner").mkdir(parents=True)
    (source / "captioner/infer.py").write_text("print('flora')\n")
    python = tmp_path / "flora-python"
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    return source, python


def flora_settings(settings_factory, tmp_path):
    source, python = make_flora(tmp_path)
    return settings_factory(
        "server",
        flora={
            "enabled": True,
            "src": str(source),
            "python": str(python),
            "model_cache": str(tmp_path / "models"),
            "hf_home": str(tmp_path / "hf"),
            "partition": "dev",
            "gpus": 1,
            "timeout_seconds": 14400,
            "profiles": {
                "agilex": {"profile": "yam", "yam32": False},
            },
        },
    )


def test_availability_checks_the_source_and_interpreter(tmp_path, settings_factory):
    settings = flora_settings(settings_factory, tmp_path)
    assert flora_annotation.available(settings) == (True, "")

    settings.flora["src"] = str(tmp_path / "missing")
    ok, reason = flora_annotation.available(settings)
    assert ok is False
    assert "源码" in reason


def test_command_runs_one_three_camera_episode_on_one_slurm_gpu(
    tmp_path, settings_factory
):
    settings = flora_settings(settings_factory, tmp_path)
    prepared = tmp_path / "prepare/episode_000004"
    prepared.mkdir(parents=True)
    for name in ("main.mp4", "left_wrist.mp4", "right_wrist.mp4"):
        (prepared / name).write_bytes(b"video")
    episode = {
        "id": "episode_000004",
        "main": str(prepared / "main.mp4"),
        "left_wrist": str(prepared / "left_wrist.mp4"),
        "right_wrist": str(prepared / "right_wrist.mp4"),
    }
    work = tmp_path / "work/episode_000004"
    argv = flora_annotation.command(
        settings,
        {
            "provider": "agilex",
            "provider_context": {},
            "note": "fold the shirt",
        },
        episode,
        work,
    )
    assert argv[:6] == [
        "srun",
        "--partition",
        "dev",
        "--gres=gpu:1",
        "--nodes=1",
        "--ntasks=1",
    ]
    assert argv[argv.index("--cpus-per-task") + 1] == "8"
    assert str(settings.flora["python"]) in argv
    assert "captioner/infer.py" in " ".join(argv)
    assert argv[argv.index("--video") + 1] == episode["main"]
    assert argv[argv.index("--left-wrist") + 1] == episode["left_wrist"]
    assert argv[argv.index("--right-wrist") + 1] == episode["right_wrist"]
    assert argv[argv.index("--profile") + 1] == "yam"
    assert argv[argv.index("--instruction") + 1] == "fold the shirt"
    assert argv[argv.index("--output") + 1] == str(
        work / "episode_000004.result.json"
    )


def test_environment_is_isolated_and_uses_flora_caches(tmp_path, settings_factory):
    settings = flora_settings(settings_factory, tmp_path)
    env = flora_annotation.environment(settings)
    assert env["FLORA_MODEL_CACHE"] == str(tmp_path / "models")
    assert env["HF_HOME"] == str(tmp_path / "hf")
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert not any(name.startswith("PIPELINE_ARGUS_") for name in env)
    assert not any(name.startswith("OPENAI_") for name in env)


def test_normalize_result_preserves_timestamps_and_incomplete_segments():
    result = flora_annotation.normalize_result(
        {
            "schema_version": "flora_demo_annotation_v1",
            "duration_s": 9.5,
            "profile": "Yam",
            "model_revisions": {"tempseg": "a", "captioner": "b"},
            "segments": [
                {
                    "start_s": 1.25,
                    "end_s": 3.75,
                    "caption": "Pick up the cup.",
                    "status": "complete",
                },
                {
                    "start_s": 5,
                    "end_s": 6,
                    "caption": "",
                    "status": "caption_failed",
                },
            ],
        },
        "episode_000004",
    )
    assert result["episode_id"] == "episode_000004"
    assert result["episode_index"] == 4
    assert result["duration_s"] == 9.5
    assert result["labels_available"] is True
    assert result["segments"] == [
        {
            "start_s": 1.25,
            "end_s": 3.75,
            "caption": "Pick up the cup.",
            "status": "complete",
        },
        {
            "start_s": 5.0,
            "end_s": 6.0,
            "caption": "",
            "status": "caption_failed",
        },
    ]
    assert result["label_failure"]["incomplete_segments"] == 1
    assert result["model"] == "Flora-TempSeg + Flora-Captioner"
    assert result["model_revisions"] == {"tempseg": "a", "captioner": "b"}


def test_normalize_result_rejects_malformed_segments():
    with pytest.raises(DomainError, match="segments"):
        flora_annotation.normalize_result(
            {"duration_s": 2, "segments": [{"start_s": "bad"}]},
            "episode_000000",
        )


def test_outputs_are_read_from_episode_sidecars(tmp_path):
    work = tmp_path / "annotation"
    work.mkdir()
    payload = {
        "duration_s": 4,
        "segments": [
            {"start_s": 0, "end_s": 4, "caption": "Move left.", "status": "complete"}
        ],
    }
    (work / "episode_000000.result.json").write_text(json.dumps(payload))

    rows = flora_annotation.outputs(work)
    assert [row["episode_id"] for row in rows] == ["episode_000000"]
    detail = flora_annotation.episode(work, "episode_000000")
    assert detail["segments"][0]["caption"] == "Move left."
