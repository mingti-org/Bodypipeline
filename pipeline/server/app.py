from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Literal
from urllib.parse import quote

from fastapi import Body, FastAPI, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt

from pipeline.shared.collections import (
    group_uploads,
    stored_collection_metadata,
    visible_collection_uploads,
)
from pipeline.shared.core import DomainError, ProcessLock, executable_path
from pipeline.shared.identity import (
    Principal,
    public_execution_record,
    require_task_access,
)
from pipeline.shared.security import install_security
from pipeline.shared.web import ListCompressionMiddleware, mount_ui

from .analyses import AnalysisRunner
from .annotations import AnnotationRunner
from .checks import QualityRunner
from .conversion_options import ConversionOptions
from .dataset_descriptions import DatasetDescriptionRequest, DatasetDescriptions
from .dataset_management import DELETION_STATES, DatasetManager
from .datasets import TARGET_FORMATS, batch_name_for_job, dataset_for_job
from .deliveries import DeliveryScanner
from .episode_validation import InvalidEpisodePolicy
from .imports import ImportService
from .ingest import IngestService
from .instruction_review import (
    ApplyInstructionsRequest,
    EpisodeSaveRequest,
    InstructionReview,
    TemplateSaveRequest,
)
from .instruction_worker import InstructionApplyRunner
from .jobs import JobRunner
from .listings import WorkbenchLists
from .providers import RuntimeCatalog
from .quality import robot_profile_blockers
from .storage import storage_path
from .subtask_review import DraftRequest, RevisionExportRequest, SubtaskReview
from .task_annotations import TemplateLibrary, TemplateRequest
from .user_sessions import SlurmIdentityBroker
from .viewer import ViewerManager
from .viewer_gateway import (
    ManualTrimRequest,
    RecycleSelection,
    ViewerTransport,
    embedded_enabled,
)

# Fields only the per-job detail view needs. They are the bulk of the listing
# response and are re-fetched from /api/jobs/{id} when a job is opened.
JOB_SUMMARY_OMIT = (
    "robot_report",
    "alignment_report",
    "episode_diagnostics",
    "source_manifest_hashes",
    "profile",
    "accepted_capture_profile_hashes",
    "doctor_report",
    "quality_gate",
    "upload_ids",
    "trim_report",
    "trim_policy",
    "merge_sources",
    "source_description",
    "source_snapshot",
)

# Keep the episode picker and Runtime checks intact; full manifests and paths
# belong to the existing upload/detail APIs, not every workbench poll.
COLLECTION_UPLOAD_FIELDS = frozenset(
    {
        "id",
        "state",
        "runtime_id",
        "robot_type",
        "robot_id",
        "capture_mode",
        "created_at",
        "total_bytes",
        "file_count",
        "import_path",
        "source_kind",
        "conversion_blockers",
    }
)


class ConversionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_format: Literal["v3.0", "v2.1"] = "v3.0"
    run_doctor: StrictBool = True
    instruction: str | None = Field(default=None, min_length=1, max_length=2000)
    conversion_options: ConversionOptions = Field(default_factory=ConversionOptions)
    invalid_episode_policy: InvalidEpisodePolicy = "fail"
    trim_idle_frames: StrictBool = False
    trim_static_intervals: StrictBool = False


class BatchConversionRequest(ConversionRequest):
    instruction: str = Field(min_length=1, max_length=2000)
    upload_ids: list[str] | None = Field(default=None, min_length=1, max_length=1000)


class ImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1, max_length=4096)
    robot_type: str | None = Field(default=None, min_length=1, max_length=64)
    runtime_id: str | None = Field(default=None, min_length=1, max_length=128)


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["local", "full"] = "local"
    episodes: list[int] | None = Field(default=None, max_length=1000)


class AnnotationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_id: str = Field(min_length=1, max_length=128)
    engine: Literal["argus", "flora"] = "argus"
    note: str | None = Field(default=None, max_length=4000)
    model: str | None = Field(default=None, min_length=1, max_length=200)
    cap_usd: float | None = Field(default=None, gt=0, le=1_000_000)
    concurrency: int | None = Field(default=None, ge=1, le=64)
    reasoning: (
        Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None
    ) = None
    sample_fps: float | None = Field(default=None, gt=0, le=10)
    inspection: Literal["auto", "all", "off"] | None = None
    field_dictionary: StrictBool | None = None
    max_tokens: int | None = Field(default=None, ge=256, le=200000)
    base_url: str | None = Field(default=None, max_length=2048)
    api_key: str | None = Field(default=None, max_length=4096)
    template_id: str | None = Field(default=None, min_length=1, max_length=160)
    template_revision: StrictInt | None = Field(default=None, ge=0)


class AnnotationResumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_url: str | None = Field(default=None, max_length=2048)
    api_key: str | None = Field(default=None, max_length=4096)


class AnnotationModelsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_url: str | None = Field(default=None, max_length=2048)
    api_key: str | None = Field(default=None, max_length=4096)


class MergeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_job_ids: list[str] = Field(min_length=2, max_length=100)
    name: str = Field(min_length=1, max_length=2000)


class VersionConversionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_job_id: str = Field(min_length=1, max_length=128)
    target_format: Literal["v3.0", "v2.1"]
    name: str | None = Field(default=None, min_length=1, max_length=2000)
    run_doctor: StrictBool = True


class SubtaskExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = Field(default=None, min_length=1, max_length=2000)


class TrimOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trim_start: StrictBool | None = None
    trim_end: StrictBool | None = None
    min_idle_seconds: float | None = Field(default=None, ge=0, le=60)
    min_active_seconds: float | None = Field(default=None, ge=0, le=60)
    padding_before_seconds: float | None = Field(default=None, ge=0, le=60)
    padding_after_seconds: float | None = Field(default=None, ge=0, le=60)
    threshold_scale: float | None = Field(default=None, ge=0.01, le=100)
    remove_fully_static: StrictBool | None = None


class TrimPreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    options: TrimOptions = Field(default_factory=TrimOptions)


class TrimRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    preview_id: str = Field(min_length=1, max_length=128)
    name: str | None = Field(default=None, min_length=1, max_length=2000)
    run_doctor: StrictBool = True


class DatasetDeletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["raw", "result"]
    id: str = Field(min_length=1, max_length=128)
    confirmation: str = Field(min_length=1, max_length=2000)


log = logging.getLogger("pipeline.server")


def registered_preview(transport: ViewerTransport, dataset_id: str) -> dict:
    row = transport.registry.lookup(f"bp/{dataset_id}")
    if row is None:
        raise DomainError(
            "预览已过期或不存在，请在管理台重新打开",
            404,
        )
    return row


def create_app(settings, background=True):
    ingest = IngestService(settings)
    runtimes = RuntimeCatalog(settings)
    imports = ImportService(settings, ingest, runtimes)
    user_sessions = SlurmIdentityBroker(settings)
    runner = JobRunner(settings, ingest, user_sessions=user_sessions)
    subtask_review = SubtaskReview(settings, ingest.store)
    instruction_review = InstructionReview(settings, ingest.store)
    instruction_apply = InstructionApplyRunner(
        settings, ingest.store, instruction_review, runner=runner
    )
    checks = QualityRunner(
        settings,
        ingest.store,
        runner.command,
        open_allocation=runner._open_allocation,
        close_allocation=runner._close_allocation,
    )
    analyses = AnalysisRunner(settings, ingest.store, runner.command)
    annotations = AnnotationRunner(
        settings, ingest.store, runtimes, runner=runner
    )
    viewer = ViewerManager(settings, ingest.store)
    datasets = DatasetManager(settings, runner, viewer)
    descriptions = DatasetDescriptions(settings, runner)
    deliveries = DeliveryScanner(settings, imports, datasets, runner)
    trims = runner.trims
    # Enqueueing consumers and marking deletions share one admission lock.
    checks.lock = analyses.lock = annotations.lock = runner.lock
    lock = ProcessLock(settings.root / "server.lock")
    stopping = asyncio.Event()

    def principal(request: Request) -> Principal:
        value = getattr(request.state, "principal", None)
        if value is None:
            raise DomainError("请先登录或提供有效凭据", 401)
        return value

    def accessible_job(
        identifier: str, request: Request, *, action: str = "操作"
    ) -> dict:
        row = ingest.store.get("job", identifier)
        require_task_access(row, principal(request), action=action)
        return row

    def check_tick():
        return runner.dataset_tick(checks, "check")

    def analysis_tick():
        return runner.dataset_tick(analyses, "analysis")

    def trim_tick():
        return runner.dataset_tick(
            trims, "trim_preview", save=trims._save_preview, run=trims._run_preview
        )

    async def loop():
        while not stopping.is_set():
            # One failing worker must not stop the shared loop: a dead loop means
            # every later job, check and analysis silently stops being picked up.
            for worker in (
                datasets.tick,
                check_tick,
                analysis_tick,
                trim_tick,
                instruction_apply.tick,
            ):
                if stopping.is_set():
                    break
                try:
                    await asyncio.to_thread(worker)
                except Exception:
                    log.exception(
                        "worker tick failed: %s", getattr(worker, "__self__", worker)
                    )
            try:
                await asyncio.wait_for(stopping.wait(), settings.poll_seconds)
            except asyncio.TimeoutError:
                pass

    async def conversion_loop():
        while not stopping.is_set():
            try:
                await asyncio.to_thread(runner.dispatch)
            except Exception:
                log.exception("conversion dispatch failed")
            try:
                await asyncio.wait_for(stopping.wait(), settings.poll_seconds)
            except asyncio.TimeoutError:
                pass

    async def viewer_loop():
        # The Visualizer is a long-lived service: start it with the Server so a
        # preview click only has to register a dataset.
        await asyncio.to_thread(viewer.autostart)
        while not stopping.is_set():
            await asyncio.to_thread(viewer.poll)
            try:
                await asyncio.wait_for(stopping.wait(), 2)
            except asyncio.TimeoutError:
                pass

    async def annotation_loop():
        # Automatic annotation is a long model-driven run. It gets its own
        # queue so a slow review never delays conversions, checks or analyses.
        while not stopping.is_set():
            try:
                await asyncio.to_thread(annotations.tick)
            except Exception:
                log.exception("annotation worker tick failed")
            try:
                await asyncio.wait_for(stopping.wait(), settings.poll_seconds)
            except asyncio.TimeoutError:
                pass

    async def delivery_loop():
        while not stopping.is_set():
            try:
                await asyncio.to_thread(deliveries.tick)
            except Exception:
                log.exception("delivery scanner tick failed")
            try:
                await asyncio.wait_for(stopping.wait(), 1)
            except asyncio.TimeoutError:
                pass

    @asynccontextmanager
    async def lifespan(app):
        lock.acquire()
        ingest.reconcile()
        runner.recover()
        checks.recover()
        analyses.recover()
        annotations.recover()
        viewer.recover()
        datasets.recover()
        trims.recover()
        instruction_apply.recover()
        if background:
            await asyncio.to_thread(lists.index.warm)
        tasks = (
            [
                asyncio.create_task(loop()),
                asyncio.create_task(conversion_loop()),
                asyncio.create_task(annotation_loop()),
                asyncio.create_task(viewer_loop()),
                asyncio.create_task(delivery_loop()),
            ]
            if background
            else []
        )
        try:
            yield
        finally:
            stopping.set()
            deliveries.stopping.set()
            runner.shutdown_requested.set()
            analyses.shutdown_requested.set()
            annotations.shutdown_requested.set()
            try:
                if tasks:
                    await asyncio.gather(*tasks)
            finally:
                try:
                    await asyncio.to_thread(runner.shutdown)
                finally:
                    try:
                        viewer.stop()
                    finally:
                        try:
                            viewer_transport.close()
                        finally:
                            try:
                                user_sessions.close_all()
                            finally:
                                lock.release()

    app = FastAPI(title="Embodied Pipeline Server", lifespan=lifespan)
    viewer_transport = ViewerTransport(settings, ingest.store, viewer.registry)
    app.state.ingest = ingest
    app.state.runner = runner
    app.state.viewer = viewer
    app.state.viewer_transport = viewer_transport
    app.state.episode_scores = viewer_transport.scores
    app.state.user_sessions = user_sessions
    app.state.imports = imports
    app.state.deliveries = deliveries
    app.state.checks = checks
    app.state.analyses = analyses
    app.state.annotations = annotations
    app.state.instructions = instruction_review
    app.state.instruction_apply = instruction_apply
    app.state.runtimes = runtimes
    app.state.datasets = datasets
    app.state.trims = trims
    install_security(
        app,
        settings,
        device_prefix="/device/",
        user_sessions=user_sessions,
    )
    if embedded_enabled(settings):
        viewer_transport.register(app, embedded=True)
    app.add_middleware(
        ListCompressionMiddleware,
        paths={
            "/api/collections",
            "/api/uploads",
            "/api/jobs",
            "/api/datasets",
            "/api/merge-candidates",
            "/api/version-conversion-candidates",
            "/api/trim-candidates",
            "/api/annotations",
            *(f"/api/listings/{name}" for name in (
                "collections", "jobs", "merge-jobs", "results", "datasets", "merge-candidates",
                "version-conversion-candidates", "trim-candidates", "annotations",
                "annotation-candidates", "collection-detail", "status", "robots",
            )),
        },
    )
    mount_ui(app, report=True)

    @app.get("/healthz")
    def live():
        return {"service": "server", "alive": True}

    @app.get("/api/config")
    def config(request: Request):
        principal = getattr(request.state, "principal", None)
        return {
            "role": "server",
            "processing_mode": settings.processing_mode,
            "auto_process": settings.auto_process,
            "max_concurrent_jobs": settings.max_concurrent_jobs,
            "robot_id": settings.robot_id,
            "profile": settings.profile(),
            "default_runtime_id": settings.default_profile or runtimes.default().id,
            "runtimes": runtimes.public(),
            "conversion_formats": [
                {"id": f, "label": "LeRobot " + f} for f in TARGET_FORMATS
            ],
            "default_target_format": "v3.0",
            "letools_available": bool(executable_path(settings.letools_bin)),
            "viewer_url": settings.viewer_origin.rstrip("/") + "/",
            "import_roots": [str(root) for root in imports.allowed_roots()],
            "current_user": principal.public() if principal else None,
            "user_login": user_sessions.enabled,
        }

    @app.post("/api/imports", status_code=201)
    def import_dataset(body: ImportRequest):
        with runner.lock, ingest.lock:
            datasets.check_import(body.path)
            return imports.import_path(
                body.path, body.robot_type, runtime_id=body.runtime_id
            )

    @app.get("/api/deliveries")
    def delivery_status():
        return deliveries.status()

    @app.post("/api/deliveries/scan", status_code=202)
    def request_delivery_scan():
        return deliveries.request()

    @app.get("/api/datasets")
    def managed_datasets():
        # Listing is advisory; enqueue revalidates under both admission locks.
        # Filesystem reads must not hold up uploads or new conversion requests.
        return datasets.listing()

    @app.post("/api/dataset-deletions", status_code=202)
    def delete_dataset(body: DatasetDeletionRequest, request: Request):
        if body.kind == "result":
            accessible_job(body.id, request, action="删除")
        return datasets.enqueue(body.kind, body.id, body.confirmation)

    @app.post("/api/dataset-deletions/{identifier}/retry", status_code=202)
    def retry_deletion(identifier: str):
        return datasets.retry(identifier)

    def authorize_upload(uid, request):
        row = ingest.store.get("upload", uid)
        device = request.state.device
        if (
            row["robot_id"] != device["robot_id"]
            or row.get("project_id", settings.project_id) != device["project_id"]
        ):
            raise DomainError("该上传不属于当前设备", 403)

    @app.get("/device/targets")
    def targets(request: Request):
        allowed = request.state.device.get("targets", list(settings.targets))
        return {
            "collection_batch_version": 1,
            "targets": [
                {"id": k, "label": v}
                for k, v in settings.targets.items()
                if k in allowed
            ]
        }

    @app.post("/device/uploads", status_code=201)
    async def upload(request: Request):
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > settings.max_manifest_bytes:
                raise DomainError("Manifest 超过上限", 413)
        try:
            data = json.loads(body)
            if not isinstance(data, dict):
                raise ValueError()
            manifest = data["manifest"]
            if not isinstance(manifest, dict):
                raise ValueError()
            runtime = runtimes.for_device(
                request.state.device, raw_format=manifest.get("raw_format")
            )
            return await asyncio.to_thread(
                ingest.create,
                manifest,
                data["target"],
                data.get("prefix", ""),
                request.state.device,
                runtime,
            )
        except (ValueError, KeyError, TypeError):
            raise DomainError("无效上传请求", 400)

    @app.get("/device/uploads/{uid}")
    def status(uid: str, request: Request):
        authorize_upload(uid, request)
        return ingest.offsets(uid)

    @app.put("/device/uploads/{uid}/files/{index}")
    async def chunk(uid: str, index: int, request: Request, offset: int):
        authorize_upload(uid, request)
        content = bytearray()
        async for part in request.stream():
            content.extend(part)
            if len(content) > settings.chunk_bytes:
                raise DomainError("分块超限", 413)
        return await asyncio.to_thread(
            ingest.chunk,
            uid,
            index,
            offset,
            request.headers.get("x-chunk-sha256", ""),
            bytes(content),
        )

    @app.post("/device/uploads/{uid}/files/{index}/reset")
    def reset(uid: str, index: int, request: Request):
        authorize_upload(uid, request)
        return ingest.reset_file(uid, index)

    @app.post("/device/uploads/{uid}/complete")
    def complete(uid: str, request: Request):
        authorize_upload(uid, request)
        return ingest.complete(uid)

    @app.get("/api/uploads")
    def uploads():
        return upload_rows(ingest.store.all("upload"))

    def upload_rows(records):
        rows = []
        for row in records:
            if row["state"] in DELETION_STATES:
                continue
            result = {**row, "conversion_blockers": []}
            blockers = result["conversion_blockers"]
            try:
                manifest = ingest.manifest(row)
                result.update(stored_collection_metadata(manifest, row))
                source_type = manifest["profile"].get("robot_type")
                result["robot_type"] = row.get("robot_type") or source_type
                try:
                    runtime = runtimes.for_stored_record(row, manifest)
                    provider = runtimes.provider(runtime)
                except DomainError:
                    # Legacy visibility: keep unknown robot types visible with
                    # actionable blockers instead of hiding their manifests.
                    blockers.extend(
                        [
                            {
                                "field": "robot_type",
                                "message": "当前服务器转换配置不适用于此机器人类型；需部署对应 Provider 和 Runtime",
                            },
                            {
                                "field": "profile_hash",
                                "message": "没有可用于此采集的已批准 Runtime/Profile",
                            },
                        ]
                    )
                    rows.append(result)
                    continue
                result["runtime_id"] = runtime.id
                result["provider"] = runtime.provider
                result["robot_type"] = runtime.robot_type
                if source_type != runtime.robot_type:
                    blockers.append(
                        {
                            "field": "robot_type",
                            "message": "采集声明的 robot_type 与设备绑定的 Runtime 不一致",
                        }
                    )
                if manifest.get("raw_format") not in runtime.raw_formats:
                    blockers.append(
                        {
                            "field": "raw_format",
                            "message": "采集原始格式不属于该 Runtime 的允许列表",
                        }
                    )
                if not runtime.accepts_profile_hash(manifest["profile_hash"]):
                    blockers.append(
                        {
                            "field": "profile_hash",
                            "message": "采集 Profile 与 Runtime 批准版本不同；需核实后加入兼容摘要",
                        }
                    )
                if manifest["capture_mode"] != "mock":
                    blockers.extend(provider.profile_blockers(runtime.profile))
            except (DomainError, OSError, ValueError, KeyError) as exc:
                blockers.append(
                    {
                        "field": "manifest",
                        "message": "采集清单不可读取或校验失败：" + str(exc),
                    }
                )
            rows.append(result)
        return rows

    @app.get("/api/collections")
    def collections(summary: bool = False):
        groups = group_uploads(visible_collection_uploads(uploads()))
        if summary:
            for group in groups:
                group["uploads"] = [
                    {k: v for k, v in row.items() if k in COLLECTION_UPLOAD_FIELDS}
                    for row in group["uploads"]
                ]
        return groups

    @app.post("/api/collections/{cid}/process", status_code=202)
    def process_collection(cid: str, body: BatchConversionRequest, request: Request):
        return job_response(
            runner.enqueue_collection(
                cid,
                **body.model_dump(),
                principal=principal(request),
            )
        )

    def job_response(row, summary=False, metadata=None):
        result = {
            **public_execution_record(row),
            "target_format": row.get("target_format", "v3.0"),
        }
        result["robot_type"] = row.get("robot_type") or row.get("profile", {}).get(
            "robot_type"
        )
        batch_name = batch_name_for_job(row)
        if batch_name:
            result["batch_name"] = batch_name
        if summary:
            # List views only render progress and names; the full reports are
            # fetched per job. Serving them here cost more than 10 MB per poll.
            for field in JOB_SUMMARY_OMIT:
                result.pop(field, None)
        if row["state"] == "BLOCKED":
            blockers = row.get("blockers")
            if blockers is None:
                try:
                    runtime = runner.runtime_for_job(row)
                    blockers = runner.catalog.provider(runtime).profile_blockers(
                        row.get("profile", runtime.profile)
                    )
                except (DomainError, ValueError, KeyError):
                    blockers = robot_profile_blockers(row.get("profile", {}))
            result["blockers"] = blockers
        try:
            result["dataset_path"] = str(dataset_for_job(settings, row) if metadata is None else metadata(row)[0])
            result["dataset_available"] = True
        except (DomainError, OSError, ValueError) as exc:
            result["dataset_available"] = False
            result["dataset_unavailable_reason"] = str(exc)
        return result

    if embedded_enabled(settings):

        def embedded_json(payload, status_code=200):
            return JSONResponse(
                payload,
                status_code=status_code,
                headers={"Cache-Control": "no-store"},
            )

        @app.get("/api/episode-scores/{dataset_id}")
        def embedded_score_status(dataset_id: str):
            row = registered_preview(viewer_transport, dataset_id)
            result = viewer_transport.scores.status(row)
            return embedded_json(result)

        @app.post("/api/episode-scores/{dataset_id}", status_code=202)
        def embedded_score_start(dataset_id: str):
            row = registered_preview(viewer_transport, dataset_id)
            result = viewer_transport.scores.start(row)
            return embedded_json(result, status_code=202)

        @app.get("/api/recycle/{dataset_id}")
        def embedded_recycle_status(dataset_id: str):
            registered_preview(viewer_transport, dataset_id)
            return embedded_json(
                viewer_transport.recycle.status(f"bp/{dataset_id}")
            )

        @app.post("/api/recycle/{dataset_id}/{action}")
        def embedded_recycle_change(
            dataset_id: str, action: str, body: RecycleSelection
        ):
            registered_preview(viewer_transport, dataset_id)
            if action not in {"trash", "restore", "purge"}:
                raise DomainError("不支持的回收站操作", 404)
            return embedded_json(
                viewer_transport.recycle.change(
                    f"bp/{dataset_id}",
                    action,
                    sorted(set(body.episodes)),
                    body.revision,
                )
            )

        @app.post("/api/manual-trims/{dataset_id}", status_code=202)
        def embedded_manual_trim_start(
            dataset_id: str, body: ManualTrimRequest, request: Request
        ):
            row = registered_preview(viewer_transport, dataset_id)
            job = trims.enqueue_manual(
                row["job_id"],
                body.episode_index,
                body.remove_ranges,
                name=body.name,
                run_doctor=body.run_doctor,
                in_place=body.in_place,
                principal=principal(request),
            )
            return embedded_json(job_response(job), status_code=202)

        def manual_trim_job(dataset_id: str, job_id: str) -> dict:
            preview = registered_preview(viewer_transport, dataset_id)
            if not job_id or any(
                character
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
                for character in job_id
            ):
                raise DomainError("裁剪任务编号无效", 404)
            try:
                job = ingest.store.get("job", job_id)
            except DomainError as exc:
                if exc.code == 404:
                    raise DomainError("裁剪任务不存在", 404) from exc
                raise
            if (
                job.get("operation") != "manual_trim"
                or job.get("source_job_id") != preview["job_id"]
            ):
                raise DomainError("裁剪任务不存在", 404)
            return job

        @app.get("/api/manual-trims/{dataset_id}/jobs/{job_id}")
        def embedded_manual_trim_status(dataset_id: str, job_id: str):
            return embedded_json(job_response(manual_trim_job(dataset_id, job_id)))

        @app.post("/api/manual-trims/{dataset_id}/jobs/{job_id}/preview")
        def embedded_manual_trim_preview(dataset_id: str, job_id: str):
            job = manual_trim_job(dataset_id, job_id)
            if job.get("state") not in {
                "COMPLETED",
                "CONVERTED",
                "SYNTHETIC_READY",
                "QUARANTINED",
            }:
                raise DomainError("裁剪尚未完成，暂时无法预览", 409)
            with runner.lock:
                opened = viewer.open(ingest.store.get("job", job_id))
            return embedded_json({"url": opened.get("url")})

    @app.get("/api/jobs")
    def jobs(summary: bool = False):
        return [
            job_response(row, summary)
            for row in ingest.store.all("job")
            if row["state"] not in DELETION_STATES
        ]

    @app.get("/api/merge-candidates")
    def merge_candidates():
        return runner.merges.candidates()

    @app.get("/api/version-conversion-candidates")
    def version_conversion_candidates():
        return runner.version_conversions.candidates()

    @app.post("/api/version-conversions", status_code=202)
    def convert_version(body: VersionConversionRequest, request: Request):
        return job_response(
            runner.version_conversions.enqueue(
                body.source_job_id,
                body.target_format,
                name=body.name,
                run_doctor=body.run_doctor,
                principal=principal(request),
            )
        )

    @app.post("/api/merges", status_code=202)
    def merge(body: MergeRequest, request: Request):
        return job_response(
            runner.merges.enqueue(
                body.source_job_ids,
                body.name,
                principal=principal(request),
            )
        )

    @app.get("/api/trim-candidates")
    def trim_candidates():
        return trims.candidates()

    @app.get("/api/jobs/{jid}/trim-policy")
    def trim_policy(jid: str):
        return trims.query_policy(jid)

    @app.get("/api/jobs/{jid}/trim-previews")
    def trim_preview_list(jid: str):
        ingest.store.get("job", jid)
        return [public_execution_record(row) for row in trims.listing(jid)]

    @app.post("/api/jobs/{jid}/trim-previews", status_code=202)
    def trim_preview_start(
        jid: str, request: Request, body: TrimPreviewRequest | None = None
    ):
        accessible_job(jid, request, action="检测")
        options = (body or TrimPreviewRequest()).options
        return public_execution_record(
            trims.enqueue_preview(
                jid,
                options.model_dump(),
                principal=principal(request),
            )
        )

    @app.get("/api/trim-previews/{pid}")
    def trim_preview(pid: str, offset: int = 0, limit: int = 200):
        return public_execution_record(trims.preview(pid, offset, limit))

    @app.post("/api/jobs/{jid}/trims", status_code=202)
    def trim_start(jid: str, body: TrimRequest, request: Request):
        accessible_job(jid, request, action="裁剪")
        return job_response(
            trims.enqueue(
                jid,
                body.preview_id,
                name=body.name,
                run_doctor=body.run_doctor,
                principal=principal(request),
            )
        )

    @app.post("/api/jobs/{jid}/manual-trims", status_code=202)
    def manual_trim_start(
        jid: str, body: ManualTrimRequest, request: Request
    ):
        accessible_job(jid, request, action="裁剪")
        return job_response(
            trims.enqueue_manual(
                jid,
                body.episode_index,
                body.remove_ranges,
                name=body.name,
                run_doctor=body.run_doctor,
                in_place=body.in_place,
                principal=principal(request),
            )
        )

    @app.post("/api/uploads/{uid}/process", status_code=202)
    def process(
        uid: str,
        request: Request,
        body: ConversionRequest | None = Body(default=None),
    ):
        options = body or ConversionRequest()
        return job_response(
            runner.enqueue(
                uid,
                **options.model_dump(),
                principal=principal(request),
            )
        )

    @app.get("/api/jobs/{jid}")
    def job(jid: str):
        return job_response(ingest.store.get("job", jid))

    @app.get("/api/jobs/{jid}/description")
    def dataset_description(jid: str):
        return descriptions.get(jid)

    @app.put("/api/jobs/{jid}/description")
    def save_dataset_description(
        jid: str, body: DatasetDescriptionRequest, request: Request
    ):
        accessible_job(jid, request, action="修改说明")
        return descriptions.save(jid, body)

    @app.post("/api/jobs/{jid}/checks", status_code=202)
    def run_check(jid: str, request: Request):
        accessible_job(jid, request, action="质检")
        # Doctor and the content analysis answer different questions about the
        # same dataset, so starting one starts both. Argus does not need letools,
        # so it is enqueued first: a broken Doctor install must not hide it.
        if settings.argus_src:
            try:
                analyses.enqueue(
                    jid,
                    mode="local",
                    principal=principal(request),
                )
            except DomainError:
                pass
        return public_execution_record(
            checks.enqueue(jid, principal=principal(request))
        )

    @app.get("/api/jobs/{jid}/checks")
    def job_checks(jid: str):
        ingest.store.get("job", jid)
        return [
            public_execution_record(row)
            for row in ingest.store.all("check")
            if row["job_id"] == jid
        ]

    @app.post("/api/jobs/{jid}/analyses", status_code=202)
    def run_analysis(
        jid: str,
        request: Request,
        body: AnalysisRequest | None = Body(default=None),
    ):
        accessible_job(jid, request, action="分析")
        options = body or AnalysisRequest()
        return public_execution_record(
            analyses.enqueue(
                jid,
                mode=options.mode,
                episodes=options.episodes,
                principal=principal(request),
            )
        )

    @app.get("/api/jobs/{jid}/analyses")
    def job_analyses(jid: str):
        ingest.store.get("job", jid)
        return [
            public_execution_record(row)
            for row in ingest.store.all("analysis")
            if row["job_id"] == jid
        ]

    @app.get("/api/analyses/{aid}")
    def analysis(aid: str):
        return public_execution_record(ingest.store.get("analysis", aid))

    @app.post("/api/analyses/{aid}/cancel")
    def cancel_analysis(aid: str, request: Request):
        require_task_access(
            ingest.store.get("analysis", aid),
            principal(request),
            action="取消",
        )
        return public_execution_record(analyses.cancel(aid))

    @app.get("/api/analyses/{aid}/budget")
    def analysis_budget(aid: str):
        row = ingest.store.get("analysis", aid)
        return analyses.budget(row["job_id"], row["episode_count"], "full")

    @app.get("/api/task-templates")
    def task_templates():
        return TemplateLibrary(ingest.store).read()

    @app.post("/api/task-templates")
    def save_task_templates(body: TemplateRequest):
        return TemplateLibrary(ingest.store).save(body)

    @app.get("/api/instructions/{dataset_id}")
    def instruction_dataset(dataset_id: str):
        return instruction_review.dataset(dataset_id)

    @app.put("/api/instructions/{dataset_id}/template")
    def instruction_template_save(
        dataset_id: str, body: TemplateSaveRequest
    ):
        return instruction_review.save_template(dataset_id, body)

    @app.get("/api/instructions/{dataset_id}/episodes/{episode}")
    def instruction_episode(dataset_id: str, episode: int):
        return instruction_review.episode(dataset_id, episode)

    @app.put("/api/instructions/{dataset_id}/episodes/{episode}")
    def instruction_episode_save(
        dataset_id: str, episode: int, body: EpisodeSaveRequest
    ):
        return instruction_review.save_episode(dataset_id, episode, body)

    @app.post(
        "/api/instructions/{dataset_id}/episodes/{episode}/apply",
        status_code=202,
    )
    def instruction_episode_apply(
        dataset_id: str,
        episode: int,
        body: ApplyInstructionsRequest,
        request: Request,
    ):
        return instruction_apply.enqueue(
            dataset_id,
            body,
            episode_indices={episode},
            principal=principal(request),
        )

    @app.post("/api/instructions/{dataset_id}/apply", status_code=202)
    def instruction_apply_start(
        dataset_id: str, body: ApplyInstructionsRequest, request: Request
    ):
        return instruction_apply.enqueue(
            dataset_id, body, principal=principal(request)
        )

    @app.get("/api/instructions/{dataset_id}/apply/{apply_id}")
    def instruction_apply_status(dataset_id: str, apply_id: str):
        return instruction_apply.status(dataset_id, apply_id)

    @app.get("/api/subtask-review/{dataset_id}/episodes/{episode}")
    def review_episode(dataset_id: str, episode: int):
        return subtask_review.episode(dataset_id, episode)

    @app.post("/api/subtask-review/{dataset_id}/episodes/{episode}")
    def review_save(dataset_id: str, episode: int, body: DraftRequest):
        return subtask_review.save(dataset_id, episode, body)

    @app.post("/api/subtask-review/{dataset_id}/exports", status_code=202)
    def review_export(
        dataset_id: str, body: RevisionExportRequest, request: Request
    ):
        accessible_job(dataset_id, request, action="导出")
        job = runner.subtask_exports.enqueue_review(
            dataset_id,
            body,
            principal=principal(request),
        )
        return {
            k: job.get(k) for k in ("id", "state", "stage", "dataset_name", "error")
        }

    @app.get("/api/subtask-review/{dataset_id}/exports/{export_id}")
    def review_export_status(dataset_id: str, export_id: str):
        subtask_review.resolve(dataset_id)
        job = ingest.store.get("job", export_id)
        if job.get("source_kind") != "lerobot_subtask_revision" or job.get("source_job_id") != dataset_id:
            raise DomainError("修订导出任务不存在", 404)
        return {k: job.get(k) for k in ("id", "state", "stage", "dataset_name", "error")}

    @app.get("/api/annotations/options")
    def annotation_options():
        return annotations.options()

    @app.get("/api/annotations/candidates")
    def annotation_candidates():
        return {"datasets": annotations.candidates()}

    @app.post("/api/annotations/models")
    def annotation_models(body: AnnotationModelsRequest):
        return annotations.models(base_url=body.base_url, api_key=body.api_key)

    @app.get("/api/annotations")
    def annotation_list():
        rows = sorted(
            ingest.store.all("annotation"),
            key=lambda row: float(row.get("created_at") or 0),
            reverse=True,
        )
        return [public_execution_record(row) for row in rows]

    @app.post("/api/annotations", status_code=202)
    def annotation_start(body: AnnotationRequest, request: Request):
        accessible_job(body.job_id, request, action="标注")
        return public_execution_record(
            annotations.enqueue(
                body.job_id,
                note=body.note,
                model=body.model,
                cap_usd=body.cap_usd,
                concurrency=body.concurrency,
                reasoning=body.reasoning,
                sample_fps=body.sample_fps,
                inspection=body.inspection,
                field_dictionary=body.field_dictionary,
                max_tokens=body.max_tokens,
                base_url=body.base_url,
                api_key=body.api_key,
                template_id=body.template_id,
                template_revision=body.template_revision,
                engine=body.engine,
                principal=principal(request),
            )
        )

    @app.get("/api/annotations/{aid}")
    def annotation(aid: str):
        return public_execution_record(ingest.store.get("annotation", aid))

    @app.get("/api/annotations/{aid}/budget")
    def annotation_budget(aid: str):
        row = ingest.store.get("annotation", aid)
        return annotations.budget(
            row["job_id"], int(row.get("input_episode_count") or 1)
        )

    @app.post("/api/annotations/{aid}/cancel")
    def annotation_cancel(aid: str, request: Request):
        require_task_access(
            ingest.store.get("annotation", aid),
            principal(request),
            action="取消",
        )
        return public_execution_record(annotations.cancel(aid))

    @app.post("/api/annotations/{aid}/resume")
    def annotation_resume(
        aid: str,
        request: Request,
        body: AnnotationResumeRequest | None = None,
    ):
        require_task_access(
            ingest.store.get("annotation", aid),
            principal(request),
            action="续跑",
        )
        return public_execution_record(
            annotations.resume(
                aid,
                base_url=body.base_url if body else None,
                api_key=body.api_key if body else None,
                principal=principal(request),
            )
        )

    @app.get("/api/annotations/{aid}/episodes")
    def annotation_episodes(aid: str):
        ingest.store.get("annotation", aid)
        return {"episodes": annotations.episodes(aid)}

    @app.get("/api/annotations/{aid}/episodes/{episode_id}")
    def annotation_episode(aid: str, episode_id: str):
        return annotations.episode(aid, episode_id)

    @app.get("/api/annotations/{aid}/episodes/{episode_id}/clips")
    def annotation_clips(aid: str, episode_id: str):
        ingest.store.get("annotation", aid)
        base = f"/api/annotations/{aid}/episodes/{episode_id}/clips"
        return {
            "clips": [
                {**row, "url": f"{base}/{quote(row['camera'], safe='')}"}
                for row in annotations.clips(aid, episode_id)
            ]
        }

    @app.get("/api/annotations/{aid}/episodes/{episode_id}/clips/{camera}")
    def annotation_clip(aid: str, episode_id: str, camera: str):
        ingest.store.get("annotation", aid)
        path = annotations.clip(aid, episode_id, camera)
        if path is None:
            raise DomainError("预览视频不存在", 404)
        # FileResponse answers byte ranges itself, which is what a <video>
        # element needs to seek inside the clip.
        return FileResponse(
            path,
            media_type="video/mp4",
            headers={"Cache-Control": "private, max-age=3600"},
        )

    @app.get("/api/annotations/{aid}/logs")
    def annotation_log_list(aid: str):
        ingest.store.get("annotation", aid)
        return {"logs": annotations.log_names(aid)}

    @app.get("/api/annotations/{aid}/logs/{name}")
    def annotation_logs(aid: str, name: str):
        return {"text": annotations.log(aid, name)}

    @app.get("/api/annotations/{aid}/export")
    def annotation_export(aid: str):
        payload = annotations.export(aid)
        return Response(
            content=json.dumps(payload, ensure_ascii=False, indent=2),
            media_type="application/json",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="annotation-{aid[:12]}.json"'
                )
            },
        )

    @app.get("/api/annotations/{aid}/subtask-export-preview")
    def subtask_export_preview(aid: str):
        return runner.subtask_exports.preview(aid)

    @app.post("/api/annotations/{aid}/subtask-exports", status_code=202)
    def subtask_export_start(
        aid: str,
        request: Request,
        body: SubtaskExportRequest | None = None,
    ):
        require_task_access(
            ingest.store.get("annotation", aid),
            principal(request),
            action="导出",
        )
        return public_execution_record(
            runner.subtask_exports.enqueue(
                aid,
                body.name if body else None,
                principal=principal(request),
            )
        )

    @app.get("/api/checks/{cid}/logs/{name}")
    def check_logs(cid: str, name: str):
        if name not in {"validate", "doctor"}:
            raise DomainError("日志不存在", 404)
        row = ingest.store.get("check", cid)
        job = ingest.store.get("job", row["job_id"])
        path = (
            storage_path(settings, job, "work", job["id"])
            / "checks"
            / cid
            / f"{name}.stderr.log"
        )
        return {
            "text": path.read_text(errors="replace")[-16000:]
            if path.exists()
            else "尚无日志"
        }

    @app.get("/api/jobs/{jid}/logs/{name}")
    def logs(jid: str, name: str):
        if name not in {
            "convert",
            "validate",
            "doctor",
            "merge",
            "source-validate",
            "environment",
            "trim",
            "trim-detect",
        }:
            raise DomainError("日志不存在", 404)
        row = ingest.store.get("job", jid)
        p = storage_path(settings, row, "work", row["id"]) / f"{name}.stderr.log"
        return {
            "text": p.read_text(errors="replace")[-16000:] if p.exists() else "尚无日志"
        }

    @app.post("/api/jobs/{jid}/viewer")
    def open_viewer(jid: str):
        with runner.lock:
            return viewer.open(ingest.store.get("job", jid))

    @app.delete("/api/jobs/{jid}/viewer")
    def close_viewer(jid: str):
        ingest.store.get("job", jid)
        return viewer.close(jid)

    @app.get("/api/viewer")
    def viewer_status():
        return viewer.status()

    @app.get("/api/uploads/{uid}/manifest")
    def manifest(uid: str):
        row = ingest.store.get("upload", uid)
        return ingest.manifest(row)

    def slim_upload_rows(records):
        return [
            {key: value for key, value in row.items() if key in COLLECTION_UPLOAD_FIELDS}
            for row in upload_rows(records)
        ]

    lists = WorkbenchLists(
        settings, ingest, runner, datasets, annotations, slim_upload_rows, job_response
    )
    app.state.lists = lists

    @app.get("/api/listings/robots")
    def list_robots():
        return {"robot_types": lists.robots()}

    @app.get("/api/listings/status")
    def list_status(robot_type: str = Query(min_length=1, max_length=128)):
        return lists.status(robot_type)

    @app.get("/api/listings/deletion-preview")
    def deletion_preview(kind: Literal["raw", "result"], id: str):
        return lists.deletion_preview(kind, id)

    @app.get("/api/listings/collection-detail")
    def collection_detail(robot_type: str = Query(min_length=1, max_length=128), id: str = Query(min_length=1, max_length=128)):
        return lists.collection_detail(robot_type, id)

    @app.get("/api/listings/{resource}")
    def list_page(
        resource: str,
        robot_type: str = Query(min_length=1, max_length=128),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=200),
        q: str = Query(default="", max_length=2000),
        kind: Literal["raw", "result"] | None = None,
    ):
        return lists.page(resource, robot_type, offset=offset, limit=limit, q=q, kind=kind)

    return app
