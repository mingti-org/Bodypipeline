"""Stable task grouping shared by collectors and servers; no database required."""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime

from .core import DomainError, digest


def dataset_name(value):
    if not isinstance(value, str):
        raise DomainError("数据集名称不能为空", 400)
    value = " ".join(unicodedata.normalize("NFKC", value).split())
    if not 1 <= len(value) <= 2000:
        raise DomainError("数据集名称须为 1–2000 个字符", 400)
    return value


def dated_batch(collection_date):
    """Validate the collector-owned date and derive its stable batch id."""
    if not isinstance(collection_date, str) or not re.fullmatch(
        r"[0-9]{8}", collection_date
    ):
        raise DomainError("采集批次日期须为 YYYYMMDD", 400)
    try:
        datetime.strptime(collection_date, "%Y%m%d")
    except ValueError as exc:
        raise DomainError("采集批次日期无效", 400) from exc
    return {
        "collection_date": collection_date,
        "batch_id": "date-" + collection_date,
    }


def batch_metadata(manifest):
    collection_date = manifest.get("collection_date")
    batch_id = manifest.get("batch_id")
    if collection_date is None and batch_id is None:
        return {}
    expected = dated_batch(collection_date)
    if batch_id != expected["batch_id"]:
        raise DomainError("采集批次标识与日期不一致", 400)
    return expected


def collection_metadata(manifest, target, prefix):
    name = dataset_name(manifest.get("dataset_name") or manifest.get("instruction"))
    scope = {
        "name": name,
        "project_id": manifest["project_id"],
        "robot_type": manifest["profile"]["robot_type"],
        "profile_hash": manifest["profile_hash"],
        "capture_mode": manifest["capture_mode"],
        "raw_format": manifest["raw_format"],
        "target": target,
        "prefix": prefix,
    }
    batch = batch_metadata(manifest)
    if batch:
        scope["batch_id"] = batch["batch_id"]
    return {"collection_id": digest(scope)[:32], "dataset_name": name, **batch}


def stored_collection_metadata(manifest, row):
    """Resolve the authoritative group, including audited offline regrouping."""
    original = collection_metadata(manifest, row["target"], row["prefix"])
    migration = row.get("collection_migration")
    if not migration:
        return original
    if (
        migration.get("version") != 1
        or migration.get("original_collection_id") != original["collection_id"]
        or migration.get("manifest_hash") != digest(manifest)
        or row.get("manifest_hash") != digest(manifest)
    ):
        raise DomainError("采集批次迁移记录与原始清单不一致")
    batch = batch_metadata(row)
    if not batch:
        raise DomainError("采集批次迁移记录缺少日期")
    result = collection_metadata({**manifest, **batch}, row["target"], row["prefix"])
    if result["collection_id"] != row.get("collection_id"):
        raise DomainError("采集批次迁移后的数据集标识不一致")
    return result


def visible_collection_uploads(rows):
    """Return episodes that have passed final validation and registration."""
    return [row for row in rows if row.get("state") == "COMMITTED"]


def group_uploads(rows):
    groups = {}
    for row in rows:
        cid = row.get("collection_id")
        if not cid:
            # A damaged legacy manifest remains visible but cannot join another task.
            cid = "unavailable-" + row["id"]
        if cid not in groups:
            groups[cid] = {
                "id": cid,
                **batch_metadata(row),
                "dataset_name": row.get("dataset_name")
                or row.get("instruction")
                or cid,
                "robot_type": row.get("robot_type"),
                "project_id": row.get("project_id"),
                "capture_mode": row.get("capture_mode"),
                "target": row.get("target"),
                "prefix": row.get("prefix", ""),
                "uploads": [],
            }
        groups[cid]["uploads"].append(row)
    for group in groups.values():
        group["uploads"].sort(key=lambda r: (r.get("created_at", 0), r["id"]))
        group["episode_count"] = len(group["uploads"])
        group["ready_count"] = sum(r["state"] == "COMMITTED" for r in group["uploads"])
        group["total_bytes"] = sum(r.get("total_bytes", 0) for r in group["uploads"])
    return list(groups.values())
