"""Shared annotation backend registry.

The queue owns persistence, cancellation, resume and export. Backends own the
engine-specific command and sidecar adaptation.
"""

from __future__ import annotations

from typing import Protocol

from pipeline.shared.core import DomainError

from . import argus_annotation, flora_annotation


class AnnotationBackend(Protocol):
    id: str

    def available(self, settings) -> tuple[bool, str]: ...

    def results(self, work) -> dict[str, dict]: ...

    def clips(self, work, episode_id, settings): ...

    def clip(self, work, episode_id, camera, settings): ...


class ArgusBackend:
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


class FloraBackend:
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
    FloraBackend.id: FloraBackend(),
}
DEFAULT_ENGINE = ArgusBackend.id


def backend_for(annotation: dict) -> AnnotationBackend:
    engine = str(annotation.get("engine") or DEFAULT_ENGINE)
    backend = BACKENDS.get(engine)
    if backend is None:
        raise DomainError(f"未知标注引擎：{engine}", 400)
    return backend


__all__ = [
    "AnnotationBackend",
    "ArgusBackend",
    "BACKENDS",
    "DEFAULT_ENGINE",
    "FloraBackend",
    "backend_for",
]
