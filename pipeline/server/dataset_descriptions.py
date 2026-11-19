"""Editable result descriptions kept separately from worker-owned job state."""

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from pipeline.shared.core import DomainError, now

from .datasets import dataset_for_job


class DatasetDescriptionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: StrictStr = Field(max_length=10000)
    revision: StrictInt = Field(ge=0)


class DatasetDescriptions:
    def __init__(self, settings, runner):
        self.settings, self.runner, self.store = settings, runner, runner.store

    def _read(self, jid):
        try:
            return self.store.get("dataset_description", jid)
        except DomainError as exc:
            if exc.code != 404:
                raise
            return {"id": jid, "description": "", "revision": 0, "updated_at": None}

    def get(self, jid):
        dataset_for_job(self.settings, self.store.get("job", jid))
        return self._read(jid)

    def save(self, jid, body):
        # Share deletion admission, and compare the latest revision under the
        # same lock so another browser cannot silently overwrite an edit.
        with self.runner.lock:
            job = self.store.get("job", jid)
            dataset_for_job(self.settings, job)
            current = self._read(jid)
            if body.revision != current["revision"]:
                raise DomainError(
                    "数据集介绍已被其他用户更新，请取消编辑以读取最新内容后再保存", 409
                )
            row = {
                "id": jid,
                "job_id": jid,
                "robot_type": job.get("robot_type")
                or job.get("profile", {}).get("robot_type"),
                "description": body.description,
                "revision": current["revision"] + 1,
                "updated_at": now(),
            }
            self.store.put("dataset_description", jid, row)
            return row
