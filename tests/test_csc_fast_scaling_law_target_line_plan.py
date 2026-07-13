from __future__ import annotations

import csv
import math
from pathlib import Path

from csc_fast_scaling_law.target_line_plan import (
    TARGET_LINE_FIELDNAMES,
    make_target_line_rows,
    write_target_line_plan,
)


ROOT = Path(__file__).resolve().parents[1]


def _read_existing_x5() -> dict[int, dict[str, str]]:
    path = ROOT / "csc_fast_scaling_law/inputs/roihu_target_line_hd64_7pt_x5_seed28_29.csv"
    with path.open(newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if int(row["seed"]) == 28]
    return {int(row["model_dim"]): row for row in rows}


def test_target_line_rows_reproduce_known_x5_iteration_counts() -> None:
    existing = _read_existing_x5()
    rows = make_target_line_rows(
        budget_multiplier=5.0,
        seeds=[28],
        model_dims=[320, 512, 896],
        run_slug="roihu-target-line-hd64-7pt-x5-seed28-29",
    )

    by_model_dim = {int(row["model_dim"]): row for row in rows}
    for model_dim in [320, 512, 896]:
        actual = by_model_dim[model_dim]
        expected = existing[model_dim]
        assert actual["num_iterations"] == expected["num_iterations"]
        assert actual["target_tokens"] == expected["target_tokens"]
        assert actual["preflight_n_scaling"] == expected["preflight_n_scaling"]
        assert math.isclose(
            float(actual["target_budget_line_residual"]),
            float(expected["target_budget_line_residual"]),
            abs_tol=1e-12,
        )


def test_target_line_rows_support_right_extension_model_dims() -> None:
    rows = make_target_line_rows(
        budget_multiplier=15.0,
        seeds=[41, 42],
        model_dims=[512, 640, 768, 896, 1024, 1152],
        run_slug="roihu-target-line-hd64-x15-extension-seed41-42",
    )

    assert len(rows) == 12
    assert [int(row["model_dim"]) for row in rows[:6]] == [512, 640, 768, 896, 1024, 1152]
    assert {row["split"] for row in rows} == {"target_line_referee"}
    assert {row["sizing_family"] for row in rows} == {"controlled_model_dim_target_line"}
    assert all(int(row["num_iterations"]) > 0 for row in rows)
    assert all(abs(float(row["target_budget_line_residual"])) < 0.002 for row in rows)


def test_write_target_line_plan_uses_stable_field_order(tmp_path: Path) -> None:
    out = tmp_path / "plan.csv"
    rows = make_target_line_rows(
        budget_multiplier=12.5,
        seeds=[41],
        model_dims=[512, 640],
        run_slug="roihu-target-line-hd64-x12p5-extension-seed41",
    )
    write_target_line_plan(out, rows)

    with out.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        body = list(reader)
    assert header == TARGET_LINE_FIELDNAMES
    assert len(body) == 2
    assert b"\r\n" not in out.read_bytes()
