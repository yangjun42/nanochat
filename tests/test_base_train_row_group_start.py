from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_base_train_exposes_train_row_group_start() -> None:
    text = (ROOT / "scripts" / "base_train.py").read_text(encoding="utf-8")

    assert "--train-row-group-start" in text
    assert "row_group_start=args.train_row_group_start" in text
    assert '"Train row-group start": args.train_row_group_start' in text


def test_base_train_exposes_optional_init_seed_for_common_random_initialization() -> None:
    text = (ROOT / "scripts" / "base_train.py").read_text(encoding="utf-8")

    assert "--init-seed" in text
    assert "effective_init_seed" in text
    assert "torch.manual_seed(effective_init_seed)" in text
    assert '"Init seed": effective_init_seed' in text


def test_run_train_array_passes_optional_train_row_group_start() -> None:
    text = (ROOT / "csc_fast_scaling_law" / "run_train_array.sbatch").read_text(encoding="utf-8")

    assert "TRAIN_ROW_GROUP_START" in text
    assert "train_row_group_args" in text
    assert "--train-row-group-start" in text


def test_run_train_array_passes_optional_init_seed() -> None:
    text = (ROOT / "csc_fast_scaling_law" / "run_train_array.sbatch").read_text(encoding="utf-8")

    assert "INIT_SEED" in text
    assert "init_seed_args" in text
    assert "--init-seed" in text


def test_run_plan_schema_documents_train_row_group_start() -> None:
    text = (ROOT / "csc_fast_scaling_law" / "inputs" / "run_plan_schema.md").read_text(encoding="utf-8")

    assert "train_row_group_start" in text
    assert "init_seed" in text
