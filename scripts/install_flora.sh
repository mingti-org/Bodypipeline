#!/usr/bin/env bash
# Validate and optionally install the isolated Flora-TempSeg + Captioner runtime.
#
# This script never downloads model weights. Set FLORA_CREATE_ENV=1 to create
# the Python 3.12 environment; download and authenticate model checkpoints
# separately after reviewing NVIDIA's research-only license.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FLORA_ROOT="${FLORA_ROOT:-$ROOT/.runtime/flora}"
FLORA_SRC="${FLORA_SRC:-/workspace/flora/Flora-Captioner}"
FLORA_PYTHON="${FLORA_PYTHON:-/workspace/flora/.venv-annotation/bin/python}"
FLORA_MODEL_CACHE="${FLORA_MODEL_CACHE:-/workspace/flora/models}"
FLORA_HF_HOME="${FLORA_HF_HOME:-/workspace/flora/huggingface}"
FLORA_TEMPSEG_MODEL="${FLORA_TEMPSEG_MODEL:-$FLORA_SRC}"
FLORA_CAPTIONER_MODEL="${FLORA_CAPTIONER_MODEL:-/workspace/flora/Flora-Captioner}"
FLORA_TEMPSEG_BASE_MODEL="${FLORA_TEMPSEG_BASE_MODEL:-$FLORA_HF_HOME/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
FLORA_REV="${FLORA_REV:-d8b755e}"

if [[ ! -f "$FLORA_SRC/captioner/infer.py" && ! -f "$FLORA_SRC/infer.py" ]]; then
  echo "Flora source is missing or incomplete: $FLORA_SRC" >&2
  echo "Set FLORA_SRC to an NVIDIA/Flora checkout at revision $FLORA_REV." >&2
  exit 1
fi

if [[ "${FLORA_CREATE_ENV:-0}" == "1" && ! -x "$FLORA_PYTHON" ]]; then
  command -v python3.12 >/dev/null || {
    echo "Python 3.12 is required to create the Flora environment." >&2
    exit 1
  }
  mkdir -p "$(dirname "$FLORA_PYTHON")/.."
  python3.12 -m venv "$(dirname "$FLORA_PYTHON")/.."
  "$FLORA_PYTHON" -m pip install --upgrade pip
  "$FLORA_PYTHON" -m pip install \
    torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cu128
  requirements="$FLORA_SRC/captioner/batch_utils/requirements.txt"
  if [[ ! -f "$requirements" ]]; then
    requirements="$FLORA_SRC/batch_utils/requirements.txt"
  fi
  if [[ ! -f "$requirements" ]]; then
    echo "Flora batch requirements are missing under: $FLORA_SRC" >&2
    exit 1
  fi
  "$FLORA_PYTHON" -m pip install -r "$requirements"
fi

if [[ ! -x "$FLORA_PYTHON" ]]; then
  echo "Flora Python interpreter is missing: $FLORA_PYTHON" >&2
  echo "Create it with FLORA_CREATE_ENV=1 or set FLORA_PYTHON." >&2
  exit 1
fi

link_model_repository() {
  local kind="$1"
  local source="$2"
  local target="$FLORA_MODEL_CACHE/$kind"
  if [[ ! -f "$source/config.json" && ! -f "$source/adapter_config.json" ]] ||
    ! compgen -G "$source/*.safetensors" >/dev/null; then
    echo "Model repository is missing model config or safetensors: $source" >&2
    return 1
  fi
  if [[ -e "$target" || -L "$target" ]]; then
    return 0
  fi
  local source_abs
  source_abs="$(cd "$source" && pwd -P)"
  ln -s "$source_abs" "$target"
}

if [[ "${FLORA_LINK_MODELS:-0}" == "1" ]]; then
  mkdir -p "$FLORA_MODEL_CACHE"
  link_model_repository tempseg "$FLORA_TEMPSEG_MODEL"
  link_model_repository captioner "$FLORA_CAPTIONER_MODEL"
  if [[ -d "$FLORA_TEMPSEG_BASE_MODEL" ]]; then
    if [[ ! -e "$FLORA_MODEL_CACHE/tempseg_base" && ! -L "$FLORA_MODEL_CACHE/tempseg_base" ]]; then
      ln -s "$(cd "$FLORA_TEMPSEG_BASE_MODEL" && pwd -P)" "$FLORA_MODEL_CACHE/tempseg_base"
    fi
  fi
fi

if [[ "${FLORA_SKIP_SYSTEM_CHECK:-0}" != "1" ]]; then
  for command in srun ffmpeg ffprobe; do
    command -v "$command" >/dev/null || {
      echo "$command is required on the Server host and PATH." >&2
      exit 1
    }
  done
fi

PYTHONPATH="$FLORA_SRC" "$FLORA_PYTHON" - <<'PY'
import av
import accelerate
import huggingface_hub
import numpy
import peft
import torch
import transformers

assert torch.__version__.startswith("2.11.")
print("Flora Python environment imports ok")
PY

if [[ "${FLORA_SKIP_MODEL_CHECK:-0}" != "1" ]]; then
  missing=0
  for kind in tempseg captioner; do
    directory="$FLORA_MODEL_CACHE/$kind"
    if [[ ! -d "$directory" ]] ||
      { [[ ! -f "$directory/config.json" ]] && [[ ! -f "$directory/adapter_config.json" ]]; } ||
      ! compgen -G "$directory/*.safetensors" >/dev/null; then
      echo "Model cache is not ready: $directory" >&2
      missing=1
    fi
  done
  base_ready=0
  for directory in \
    "$FLORA_MODEL_CACHE/tempseg_base" \
    "$FLORA_HF_HOME"/hub/models--Qwen--Qwen3.5-9B/snapshots/*; do
    [[ -d "$directory" ]] || continue
    [[ -f "$directory/config.json" ]] || continue
    compgen -G "$directory/*.safetensors" >/dev/null || continue
    base_ready=1
    break
  done
  if (( ! base_ready )); then
    echo "Missing Qwen3.5-9B base model in $FLORA_MODEL_CACHE/tempseg_base" >&2
    echo "or the Hugging Face cache under $FLORA_HF_HOME." >&2
    missing=1
  fi
  if (( missing )); then
    echo "Flora weights are incomplete. Download them separately; this script never fetches them." >&2
    exit 1
  fi
  MODEL_CACHE="$FLORA_MODEL_CACHE" HF_HOME_CHECK="$FLORA_HF_HOME" "$FLORA_PYTHON" - <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(os.environ["MODEL_CACHE"])

def ready(path):
    if not path.is_dir():
        return False, "directory is missing"
    if not (path / "config.json").is_file() and not (path / "adapter_config.json").is_file():
        return False, "model config is missing"
    files = list(path.glob("*.safetensors"))
    if not files or any(not file.is_file() or file.stat().st_size <= 0 for file in files):
        return False, "safetensors are missing or empty"
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text())
            shards = {path / str(value) for value in (index.get("weight_map") or {}).values()}
        except (OSError, TypeError, ValueError):
            return False, "safetensors index is invalid"
        if not shards or any(not file.is_file() or file.stat().st_size <= 0 for file in shards):
            return False, "one or more indexed safetensors shards are missing"
        total = (index.get("metadata") or {}).get("total_size")
        if type(total) is int and sum(file.stat().st_size for file in shards) != total:
            return False, "safetensors shards are incomplete"
    return True, ""

missing = False
for kind in ("tempseg", "captioner"):
    ok, why = ready(root / kind)
    if not ok:
        print(f"Model cache is not ready: {root / kind}: {why}", file=__import__("sys").stderr)
        missing = True

base_candidates = [root / "tempseg_base"]
hf = Path(os.environ["HF_HOME_CHECK"])
base_candidates.extend((hf / "hub/models--Qwen--Qwen3.5-9B/snapshots").glob("*"))
if not any(ready(path)[0] for path in base_candidates):
    print(
        f"Missing complete Qwen3.5-9B base model in {root / 'tempseg_base'} "
        f"or the Hugging Face cache under {hf}.",
        file=__import__("sys").stderr,
    )
    missing = True
if missing:
    print(
        "Flora weights are incomplete. Download them separately; "
        "this script never fetches them.",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
fi

mkdir -p "$FLORA_HF_HOME"
cat <<EOF

Flora annotation backend is ready.

  "flora": {
    "enabled": true,
    "src": "$FLORA_SRC",
    "python": "$FLORA_PYTHON",
    "srun": "srun",
    "ffmpeg": "ffmpeg",
    "ffprobe": "ffprobe",
    "model_cache": "$FLORA_MODEL_CACHE",
    "hf_home": "$FLORA_HF_HOME",
    "hf_endpoint": "https://hf-mirror.com",
    "partition": "dev",
    "gpus": 1,
    "cpus_per_task": 8,
    "timeout_seconds": 14400,
    "profiles": {
      "<provider>": {
        "profile": "yam",
        "yam32": false,
        "camera_roles": {
          "main": "observation.images.front",
          "left_wrist": "observation.images.left_wrist",
          "right_wrist": "observation.images.right_wrist"
        }
      }
    }
  }

The selected Provider must explicitly declare one camera for role "main";
wrist roles are optional. The output is for non-commercial research and
evaluation under NVIDIA's Flora model license. Verify that the Server account
can read all configured paths and submit srun jobs before enabling this block.
EOF
