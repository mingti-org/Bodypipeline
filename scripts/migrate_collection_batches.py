#!/usr/bin/env python3
"""Run with the deployed Server environment, after stopping its service."""

import argparse
import json
from pathlib import Path

from pipeline.server.collection_migration import apply_split, plan_split, rollback_split
from pipeline.server.ingest import IngestService
from pipeline.shared.config import Settings
from pipeline.shared.core import ProcessLock, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--action", choices=["plan", "apply", "rollback"], required=True
    )
    parser.add_argument("--assignments", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--journal", type=Path)
    args = parser.parse_args()
    settings = Settings.load("server", args.config)
    lock = ProcessLock(settings.root / "server.lock")
    lock.acquire()
    try:
        ingest = IngestService(settings)
        if args.action == "plan":
            if not args.assignments or not args.plan:
                parser.error("plan requires --assignments and --plan")
            plan = plan_split(ingest, json.loads(args.assignments.read_text()))
            atomic_json(args.plan, plan)
            print(
                json.dumps(
                    {
                        "episodes": len(plan["entries"]),
                        "plan_hash": plan["plan_hash"],
                        "collections": sorted(
                            {e["after"]["collection_id"] for e in plan["entries"]}
                        ),
                    }
                )
            )
        elif args.action == "apply":
            if not args.plan or not args.journal:
                parser.error("apply requires --plan and --journal")
            result = apply_split(
                ingest, json.loads(args.plan.read_text()), args.journal
            )
            print(result["state"])
        else:
            if not args.journal:
                parser.error("rollback requires --journal")
            print(rollback_split(ingest, args.journal)["state"])
    finally:
        lock.release()


if __name__ == "__main__":
    main()
