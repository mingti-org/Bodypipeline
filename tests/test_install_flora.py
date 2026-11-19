"""Smoke tests for the non-destructive Flora deployment preflight."""

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts/install_flora.sh"


def run_installer(
    tmp_path,
    *,
    complete_models=True,
    model_repository_root=False,
    base_in_hf_cache=False,
    link_models=False,
    **overrides,
):
    source = tmp_path / "flora"
    source.mkdir(parents=True)
    if model_repository_root:
        (source / "infer.py").write_text("")
        (source / "config.json").write_text("{}")
        (source / "adapter_model.safetensors").write_bytes(b"weights")
    else:
        (source / "captioner").mkdir()
        (source / "tempseg").mkdir()
        (source / "captioner/infer.py").write_text("")
        (source / "tempseg/infer.py").write_text("")
    captioner_model = tmp_path / "Flora-Captioner"
    captioner_model.mkdir()
    (captioner_model / "config.json").write_text("{}")
    (captioner_model / "model.safetensors").write_bytes(b"weights")
    python = tmp_path / "flora-python"
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    models = tmp_path / "models"
    if not link_models:
        for kind in ("tempseg", "captioner"):
            directory = models / kind
            directory.mkdir(parents=True, exist_ok=True)
            if not complete_models:
                continue
            (directory / "config.json").write_text("{}")
            (directory / "model.safetensors").write_bytes(b"weights")
    if complete_models:
        if base_in_hf_cache:
            base = (
                tmp_path
                / "hf/hub/models--Qwen--Qwen3.5-9B/snapshots/"
                "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
            )
        else:
            base = models / "tempseg_base"
        base.mkdir(parents=True, exist_ok=True)
        (base / "config.json").write_text("{}")
        (base / "model.safetensors").write_bytes(b"weights")
    env = {
        **os.environ,
        "FLORA_SRC": str(source),
        "FLORA_PYTHON": str(python),
        "FLORA_MODEL_CACHE": str(models),
        "FLORA_HF_HOME": str(tmp_path / "hf"),
        "FLORA_SKIP_SYSTEM_CHECK": "1",
    }
    if link_models:
        env.update(
            {
                "FLORA_LINK_MODELS": "1",
                "FLORA_TEMPSEG_MODEL": str(source),
                "FLORA_CAPTIONER_MODEL": str(captioner_model),
            }
        )
    env.update(overrides)
    return subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
    )


def test_installer_reports_ready_config_without_downloading_weights(tmp_path):
    completed = run_installer(tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert "Flora annotation backend is ready" in completed.stdout
    assert '"gpus": 1' in completed.stdout
    assert (tmp_path / "hf").is_dir()


def test_installer_rejects_missing_source(tmp_path):
    completed = run_installer(tmp_path, FLORA_SRC=str(tmp_path / "missing"))
    assert completed.returncode != 0
    assert "source is missing or incomplete" in completed.stderr


def test_installer_accepts_a_model_repository_root_infer(tmp_path):
    completed = run_installer(tmp_path, model_repository_root=True)
    assert completed.returncode == 0, completed.stderr
    assert "Flora annotation backend is ready" in completed.stdout


def test_installer_accepts_the_qwen_base_in_the_huggingface_cache(tmp_path):
    completed = run_installer(tmp_path, base_in_hf_cache=True)
    assert completed.returncode == 0, completed.stderr
    assert "Flora annotation backend is ready" in completed.stdout


def test_installer_links_preexisting_model_repository_directories(tmp_path):
    completed = run_installer(
        tmp_path,
        base_in_hf_cache=True,
        link_models=True,
        model_repository_root=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "models/tempseg").is_symlink()
    assert (tmp_path / "models/captioner").is_symlink()
    assert (tmp_path / "models/tempseg").resolve() == (tmp_path / "flora").resolve()
    assert (tmp_path / "models/captioner").resolve() == (
        tmp_path / "Flora-Captioner"
    ).resolve()


def test_installer_rejects_incomplete_model_cache(tmp_path):
    models = tmp_path / "models"
    for kind in ("tempseg", "captioner", "tempseg_base"):
        (models / kind).mkdir(parents=True, exist_ok=True)
    completed = run_installer(
        tmp_path,
        complete_models=False,
        FLORA_MODEL_CACHE=str(models),
    )
    assert completed.returncode != 0
    assert "weights are incomplete" in completed.stderr
