#!/usr/bin/env python3
"""Create a fully synthetic campaign fixture for contract smoke tests."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

from csc_fast_scaling_law.campaign_contract import sha256_file


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def create_fixture(root: Path) -> Path:
    root = root.expanduser().resolve()
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    checkout = root / "synthetic_checkout"
    checkout.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "config", "user.name", "Synthetic Fixture"], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "synthetic@example.invalid"],
        check=True,
    )
    (checkout / "code.txt").write_text("synthetic fixture code\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "code.txt"], check=True)
    subprocess.run(["git", "-C", str(checkout), "commit", "-q", "-m", "synthetic"], check=True)
    code_commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    plan = inputs / "plan.csv"
    rows = [
        {
            "task_id": "synthetic-a",
            "init_seed": "1",
            "data_seed": "11",
            "schedule_id": "fake-schedule",
            "gpu_count": "1",
            "target_state": "source",
        },
        {
            "task_id": "synthetic-b",
            "init_seed": "2",
            "data_seed": "12",
            "schedule_id": "fake-schedule",
            "gpu_count": "1",
            "target_state": "source",
        },
    ]
    with plan.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    files = {
        "plan": plan,
        "plan_frozen": inputs / "PLAN_FROZEN",
        "data_manifest": inputs / "data.json",
        "validation_manifest": inputs / "validation.json",
        "tokenizer_manifest": inputs / "tokenizer.json",
        "tokenizer_consistency_receipt": inputs / "tokenizer_consistency_receipt.json",
        "environment_lock": inputs / "environment.lock",
        "schedule_manifest": inputs / "schedule.json",
        "decision_rules": inputs / "decision_rules.json",
        "data_preflight_receipt": inputs / "data_preflight_receipt.json",
        "text_overlap_audit": inputs / "text_overlap_audit.json",
        "data_capacity_receipt": inputs / "data_capacity_receipt.json",
    }
    for name, path in files.items():
        if name != "plan":
            write_json(path, {"synthetic": True, "kind": name})

    runner = root / "synthetic_runner.py"
    runner.write_text(
        """#!/usr/bin/env python3
import json
import os
import pathlib
import sys

prepared = json.loads(pathlib.Path(sys.argv[1]).read_text())
root = pathlib.Path(prepared[\"staging_dir\"])
task_id = prepared[\"task_id\"]
checkpoint = root / \"work\" / \"base_checkpoints\" / task_id
checkpoint.mkdir(parents=True, exist_ok=True)
(checkpoint / \"model.pt\").write_text(\"synthetic checkpoint; not a model\\n\")
(checkpoint / \"meta.json\").write_text(json.dumps({\"synthetic\": True}) + \"\\n\")
metrics = root / \"metrics\" / \"final.json\"
metrics.parent.mkdir(parents=True, exist_ok=True)
metrics.write_text(json.dumps({\"val_bpb\": 1.0, \"N_scaling\": 1000, \"D_actual\": 2000}) + \"\\n\")
""",
        encoding="utf-8",
    )
    runner.chmod(0o755)
    files["task_runner"] = runner

    campaign = {
        "schema_version": "nanochat-campaign-v1",
        "campaign_id": "synthetic-example",
        "code": {"commit": code_commit, "dirty": False},
        "files": {
            name: {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}
            for name, path in files.items()
        },
        "artifacts": {
            "root": "artifacts",
            "required": [
                {
                    "name": "checkpoint",
                    "glob": "work/base_checkpoints/{task_id}/model.pt",
                    "min_count": 1,
                    "max_count": 1,
                },
                {
                    "name": "checkpoint_meta",
                    "glob": "work/base_checkpoints/{task_id}/meta.json",
                    "min_count": 1,
                    "max_count": 1,
                },
                {
                    "name": "metrics",
                    "glob": "metrics/final.json",
                    "min_count": 1,
                    "max_count": 1,
                },
            ],
            "forbid_globs": ["work/**/optim_*.pt"],
        },
        "reducer": {
            "metric_json": "metrics/final.json",
            "required_fields": ["val_bpb", "N_scaling", "D_actual"],
        },
    }
    campaign_path = root / "campaign.json"
    write_json(campaign_path, campaign)
    return campaign_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(create_fixture(args.output))


if __name__ == "__main__":
    main()
