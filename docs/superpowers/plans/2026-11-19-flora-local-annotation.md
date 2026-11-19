# Flora Local Annotation Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a server-managed Flora-TempSeg + Flora-Captioner annotation engine beside Argus without duplicating the annotation queue or touching the Viewer.

**Architecture:** `AnnotationRunner` remains the shared queue, cancellation, resume, persistence and export owner. A backend contract isolates engine-specific availability, validation, process construction, progress and result adaptation. Flora runs one synchronized LeRobot Episode per Slurm GPU allocation, normalizes its timestamped captions into the existing Episode result shape, and is available only through trusted server settings and Provider context.

**Tech Stack:** Python 3.12, FastAPI, pytest, JSON/JSONL, LeRobot v2.1/v3.0 metadata, FFmpeg, Slurm, Flora `captioner/infer.py`, vanilla JavaScript.

**Spec:** The approved integration design in the current task; no separate design document was requested.

## Global Constraints

- First release uses Flora-TempSeg and Flora-Captioner only. Do not invoke Flora-Rewarder.
- Do not modify `pipeline/server/viewer_ui/*`, Viewer patches, `viewer.py`, or Viewer annotation bridge files.
- Do not branch the Server page or API on robot model names. Engine capability and camera behavior come from Provider configuration.
- Flora accepts at most 120 seconds, 7200 frames and 256 MiB per video; all supplied cameras must have synchronized timestamps, FPS and frame counts.
- Run one Episode per allocation with `partition=dev` and `gres=gpu:1` by default.
- Free analysis is supported first. Shared fixed-task templates remain Argus-only while selected.
- Model weights, videos, API keys, tokens and generated outputs are never committed.
- Existing automatic annotation records without `engine` remain Argus records.

---

### Task 1: Define the annotation backend contract and engine selection

**Files:**
- Create: `pipeline/server/annotation_backends.py`
- Modify: `pipeline/server/annotations.py`
- Modify: `pipeline/server/app.py`
- Test: `tests/test_annotations.py`

**Interfaces:**
- Produces: `AnnotationBackend` protocol with `id`, `available(settings)`, `validate(annotation, provider_context)`, `command(settings, annotation, data, work)`, `environment(settings, annotation, credentials)`, `stage_from_log(text, current)`, `progress(work)`, `outputs(work)`, `episode(work, episode_id)`, `clips(work, episode_id)`, `clip(work, episode_id, camera)`, `log_names(work)`, `read_log(work, name)` and `cost(work)`.
- Produces: `backend_for(annotation)` with `engine` defaulting to `"argus"`.
- Produces: `AnnotationEngineRegistry.options()` returning per-engine availability and Argus-specific options.
- Consumes: existing `AnnotationRunner.enqueue()` request shape plus optional `engine="argus"|"flora"`.

- [ ] **Step 1: Write failing contract tests** proving that an annotation without `engine` dispatches to Argus, `engine="flora"` dispatches to Flora, an unknown engine is rejected, and an unavailable selected engine does not block the other engine's options.
- [ ] **Step 2: Run `pytest tests/test_annotations.py -q`** and confirm failures name missing engine selection and backend dispatch.
- [ ] **Step 3: Add the protocol, registry and minimal Argus adapter** while preserving all existing Argus behavior and records.
- [ ] **Step 4: Add `engine` to the API request model and enqueue path**, validate its literal value, and persist the selected engine.
- [ ] **Step 5: Run `pytest tests/test_annotations.py -q`** and keep the complete existing suite green.
- [ ] **Step 6: Commit** with `feat(server): add annotation backend selection`.

### Task 2: Add Flora settings, availability and result adaptation

**Files:**
- Create: `pipeline/server/flora_annotation.py`
- Modify: `pipeline/shared/config.py`
- Test: `tests/test_flora_annotation.py`

**Interfaces:**
- Produces: `available(settings) -> tuple[bool, str]`.
- Produces: `command(settings, annotation, media, work) -> list[str]`.
- Produces: `environment(settings, annotation) -> dict`.
- Produces: `outputs(work) -> list[dict]` and `episode(work, episode_id) -> dict|None`.
- Produces: `normalize_result(payload, episode_id) -> dict` in the shared result shape used by the annotation page.

Flora settings live under one JSON object named `flora`:

```json
{
  "enabled": true,
  "src": "/workspace/flora/Flora-TempSeg",
  "python": "/workspace/flora/.venv-annotation/bin/python",
  "model_cache": "/workspace/flora/models",
  "hf_home": "/workspace/flora/huggingface",
  "partition": "dev",
  "gpus": 1,
  "timeout_seconds": 14400,
  "profiles": {
    "agilex": {"profile": "yam", "yam32": false},
    "g1_sonic": {"profile": "dex"},
    "arx_kinesthetic": {"profile": "dex"},
    "arx_direct_teaching": {"profile": "dex"}
  }
}
```

- [ ] **Step 1: Write failing tests** for disabled settings, missing executable/source, provider-to-profile lookup, command arguments, environment isolation, complete/incomplete status normalization, and malformed output rejection.
- [ ] **Step 2: Run `pytest tests/test_flora_annotation.py -q`** and confirm the module/interfaces are missing.
- [ ] **Step 3: Implement settings validation and Flora availability** with no model branch in API handlers.
- [ ] **Step 4: Implement command/environment construction** targeting `captioner/infer.py` and the isolated annotation environment.
- [ ] **Step 5: Implement exact result adaptation** from `{start_s,end_s,caption,status}` to the existing `{t_s,end_s,verb_class}` export representation while preserving incomplete status and model metadata.
- [ ] **Step 6: Run the focused tests and `pytest tests/test_annotations.py -q`**, then commit with `feat(server): add Flora annotation backend`.

### Task 3: Prepare LeRobot Episodes and safe Flora media slices

**Files:**
- Create: `pipeline/server/flora_media_worker.py`
- Create: `tests/fixtures/flora_media_scenario.py`
- Test: `tests/test_flora_media.py`

**Interfaces:**
- Produces: `load_episode_manifest(dataset: Path) -> list[EpisodeMedia]` for LeRobot v2.1 and v3.0.
- Produces: `prepare_episode(dataset, episode, destination, ffmpeg="ffmpeg", ffprobe="ffprobe") -> dict` returning synchronized main/left/right slice paths plus duration and source hashes.
- Produces CLI `python -m pipeline.server.flora_media_worker --dataset PATH --output DIR`.

- [ ] **Step 1: Write failing tests** using synthetic v2.1 and v3.0 metadata for camera mapping, episode selection, interval validation and output paths.
- [ ] **Step 2: Add a failing integration-style test** with fake `ffprobe`/`ffmpeg` executables proving that an Episode over 120 seconds is split into bounded slices and that each slice gets an absolute offset.
- [ ] **Step 3: Run `pytest tests/test_flora_media.py -q`** and confirm failures are due to missing manifest/media preparation.
- [ ] **Step 4: Implement metadata-only episode discovery** using `meta/info.json` and `meta/episodes.jsonl` / `meta/episodes/*.parquet` without modifying the source dataset.
- [ ] **Step 5: Implement FFmpeg extraction** with stream-copy first, bounded validation and explicit re-encode fallback; verify all camera views have matching duration before model execution.
- [ ] **Step 6: Implement slice time offsets and deterministic output manifests**, then run focused tests and commit with `feat(server): prepare LeRobot media for Flora`.

### Task 4: Integrate Flora into the shared queue and Slurm lifecycle

**Files:**
- Modify: `pipeline/server/annotations.py`
- Modify: `pipeline/server/flora_annotation.py`
- Test: `tests/test_annotations.py`
- Test: `tests/test_flora_annotation.py`

- [ ] **Step 1: Write failing tests** proving Flora records use no Argus credentials or USD budget, set `episode_count` from the dataset, skip Argus templates, and invoke a mock Flora process.
- [ ] **Step 2: Write failing Slurm tests** proving the command requests one GPU, one Episode per allocation, cancellation terminates the process group, and completed Episode outputs survive resume.
- [ ] **Step 3: Run `pytest tests/test_annotations.py tests/test_flora_annotation.py -q`** and confirm the expected failures.
- [ ] **Step 4: Add engine-aware enqueue and resume validation**, keeping legacy rows on Argus and disabling shared templates for Flora.
- [ ] **Step 5: Add per-Episode Flora orchestration and progress harvesting**, reusing the existing queue states and cancellation event.
- [ ] **Step 6: Run focused tests and commit** with `feat(server): run Flora annotations on Slurm GPUs`.

### Task 5: Add engine-aware automatic annotation UI

**Files:**
- Modify: `pipeline/web/index.html`
- Modify: `pipeline/web/annotations.js`
- Test: `tests/test_server_api.py` or a focused static API contract test

- [ ] **Step 1: Add a failing API/UI contract test** requiring the options payload to expose `engines` and the annotation records to expose `engine`.
- [ ] **Step 2: Run the focused test and confirm it fails.**
- [ ] **Step 3: Add the engine selector** with `Argus 云端` and `Flora 本地`.
- [ ] **Step 4: Hide API Key, base URL, cap, token and Argus quality controls when Flora is selected; disable shared task templates and show the Slurm/model readiness reason.**
- [ ] **Step 5: Show the selected engine in the queue, detail, logs and export state without changing Viewer files.**
- [ ] **Step 6: Run Python and JavaScript tests and commit** with `feat(web): choose automatic annotation engine`.

### Task 6: Add installation and deployment support

**Files:**
- Create: `scripts/install_flora.sh`
- Modify: `README.md`
- Modify: `configs/server.json` only with disabled default settings

- [ ] **Step 1: Add a shell-level smoke test** using a temporary fake Flora tree to verify the script refuses missing source, missing Python and incomplete models with actionable messages.
- [ ] **Step 2: Run the smoke test and confirm it fails before the script exists.**
- [ ] **Step 3: Implement source/checkpoint validation, isolated Python 3.12 environment setup and CPU `--validate-only` guidance.**
- [ ] **Step 4: Document Slurm, model-cache, non-commercial research-license and exact operational commands without embedding secrets.**
- [ ] **Step 5: Run the smoke test and commit** with `docs(deploy): install Flora annotation backend`.

### Task 7: End-to-end verification

**Files:**
- Test: all touched test modules

- [ ] **Step 1: Run `pytest tests/test_annotations.py tests/test_flora_annotation.py tests/test_flora_media.py -q`.**
- [ ] **Step 2: Run `ruff check pipeline/server/annotations.py pipeline/server/annotation_backends.py pipeline/server/flora_annotation.py pipeline/server/flora_media_worker.py pipeline/shared/config.py tests/test_annotations.py tests/test_flora_annotation.py tests/test_flora_media.py`.**
- [ ] **Step 3: Run the existing Server API, template, Subtask and Viewer isolation suites.**
- [ ] **Step 4: Inspect `git diff --name-only` and confirm no Viewer file changed.**
- [ ] **Step 5: Report any unverified H800 GPU requirement and the exact deployment command; do not deploy or push unless separately requested.**
