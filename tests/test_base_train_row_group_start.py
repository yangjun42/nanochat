from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_base_train_exposes_train_row_group_start() -> None:
    text = (ROOT / "scripts" / "base_train.py").read_text(encoding="utf-8")

    assert "--train-row-group-start" in text
    assert "row_group_start=args.train_row_group_start" in text
    assert '"Train row-group start": args.train_row_group_start' in text


def test_run_train_array_passes_optional_train_row_group_start() -> None:
    text = (ROOT / "csc_fast_scaling_law" / "run_train_array.sbatch").read_text(encoding="utf-8")

    assert "TRAIN_ROW_GROUP_START" in text
    assert "train_row_group_args" in text
    assert "--train-row-group-start" in text


def test_run_plan_schema_documents_train_row_group_start() -> None:
    text = (ROOT / "csc_fast_scaling_law" / "inputs" / "run_plan_schema.md").read_text(encoding="utf-8")

    assert "train_row_group_start" in text
