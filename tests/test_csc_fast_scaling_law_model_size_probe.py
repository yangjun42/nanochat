from __future__ import annotations

import pytest

from csc_fast_scaling_law.model_size_probe import (
    make_formula_model_size_row,
    make_model_size_row,
    make_model_size_table,
    rank_geometric_triplets,
    select_geometric_triplet,
)


def test_make_model_size_row_matches_base_train_architecture_rounding() -> None:
    row = make_model_size_row(
        depth=1,
        aspect_ratio=64,
        head_dim=128,
        max_seq_len=2048,
        vocab_size=1024,
        window_pattern="L",
    )

    assert row["depth"] == 1
    assert row["aspect_ratio"] == 64
    assert row["head_dim"] == 128
    assert row["n_embd"] == 128
    assert row["n_head"] == 1
    assert row["N_total"] > row["N_scaling"] > 0
    assert row["flops_per_token_est"] > 6 * row["N_scaling"]


def test_formula_model_size_row_matches_meta_model_counts() -> None:
    meta_row = make_model_size_row(
        depth=7,
        aspect_ratio=72,
        head_dim=128,
        max_seq_len=2048,
        vocab_size=32768,
        window_pattern="L",
    )

    formula_row = make_formula_model_size_row(
        depth=7,
        aspect_ratio=72,
        head_dim=128,
        max_seq_len=2048,
        vocab_size=32768,
        window_pattern="L",
    )

    assert formula_row == meta_row


def test_model_size_table_can_use_formula_counts() -> None:
    rows = make_model_size_table(
        depths=[2, 4, 7],
        aspect_ratios=[72],
        head_dim=128,
        use_formula_counts=True,
    )

    assert [row["N_scaling"] for row in rows] == [9_961_496, 19_660_872, 38_797_504]


def test_model_size_table_can_scan_multiple_head_dims() -> None:
    rows = make_model_size_table(
        depths=[1],
        aspect_ratios=[64],
        head_dims=[64, 128],
        use_formula_counts=True,
    )

    assert [row["head_dim"] for row in rows] == [64, 128]
    assert [row["n_embd"] for row in rows] == [64, 128]


def test_select_geometric_triplet_prefers_even_measured_n_scaling_ratios() -> None:
    rows = [
        {"depth": 1, "aspect_ratio": 64, "N_scaling": 100.0},
        {"depth": 2, "aspect_ratio": 64, "N_scaling": 205.0},
        {"depth": 4, "aspect_ratio": 64, "N_scaling": 415.0},
        {"depth": 1, "aspect_ratio": 96, "N_scaling": 100.0},
        {"depth": 3, "aspect_ratio": 96, "N_scaling": 200.0},
        {"depth": 6, "aspect_ratio": 96, "N_scaling": 400.0},
    ]

    selected = select_geometric_triplet(rows)

    assert [row["depth"] for row in selected] == [1, 3, 6]
    assert [row["aspect_ratio"] for row in selected] == [96, 96, 96]


def test_select_geometric_triplet_prefers_fixed_controls_over_tiny_spacing_gain() -> None:
    rows = [
        {"depth": 2, "aspect_ratio": 48, "head_dim": 128, "N_scaling": 100.0},
        {"depth": 3, "aspect_ratio": 48, "head_dim": 128, "N_scaling": 230.0},
        {"depth": 7, "aspect_ratio": 48, "head_dim": 128, "N_scaling": 520.0},
        {"depth": 2, "aspect_ratio": 64, "head_dim": 128, "N_scaling": 100.0},
        {"depth": 5, "aspect_ratio": 80, "head_dim": 128, "N_scaling": 220.0},
        {"depth": 8, "aspect_ratio": 96, "head_dim": 128, "N_scaling": 484.0},
    ]

    selected = select_geometric_triplet(rows)

    assert [row["aspect_ratio"] for row in selected] == [48, 48, 48]


def test_rank_geometric_triplets_returns_explainable_top_candidates() -> None:
    rows = [
        {"depth": 2, "aspect_ratio": 48, "head_dim": 128, "N_scaling": 100.0},
        {"depth": 3, "aspect_ratio": 48, "head_dim": 128, "N_scaling": 230.0},
        {"depth": 7, "aspect_ratio": 48, "head_dim": 128, "N_scaling": 520.0},
        {"depth": 2, "aspect_ratio": 64, "head_dim": 128, "N_scaling": 100.0},
        {"depth": 5, "aspect_ratio": 80, "head_dim": 128, "N_scaling": 220.0},
        {"depth": 8, "aspect_ratio": 96, "head_dim": 128, "N_scaling": 484.0},
    ]

    ranked = rank_geometric_triplets(rows, top_k=2)

    assert len(ranked) == 2
    assert ranked[0]["rank"] == 1
    assert ranked[0]["control_changes"] == 0
    assert ranked[0]["depths"] == "2,3,7"
    assert ranked[0]["aspect_ratios"] == "48,48,48"
    assert ranked[0]["n_values"] == "100,230,520"
    assert ranked[0]["n_ratio_01"] == 2.3
    assert ranked[0]["n_ratio_12"] == 520.0 / 230.0
    assert ranked[0]["spacing_error"] > 0
    assert ranked[1]["rank"] == 2


def test_rank_geometric_triplets_reports_exact_integer_geometry() -> None:
    rows = [
        {"depth": 2, "aspect_ratio": 72, "head_dim": 128, "N_scaling": 9_961_496},
        {"depth": 4, "aspect_ratio": 72, "head_dim": 128, "N_scaling": 19_660_872},
        {"depth": 7, "aspect_ratio": 72, "head_dim": 128, "N_scaling": 38_797_504},
    ]

    ranked = rank_geometric_triplets(rows, top_k=1)

    assert ranked[0]["is_exact_geometric"] is False
    assert ranked[0]["geometric_cross_product_residual"] == 68_706_894_400


def test_rank_geometric_triplets_marks_exact_integer_triples() -> None:
    rows = [
        {"depth": 1, "aspect_ratio": 64, "head_dim": 128, "N_scaling": 100},
        {"depth": 2, "aspect_ratio": 64, "head_dim": 128, "N_scaling": 200},
        {"depth": 4, "aspect_ratio": 64, "head_dim": 128, "N_scaling": 400},
    ]

    ranked = rank_geometric_triplets(rows, top_k=1)

    assert ranked[0]["is_exact_geometric"] is True
    assert ranked[0]["geometric_cross_product_residual"] == 0


def test_rank_geometric_triplets_can_filter_to_exact_geometry() -> None:
    rows = [
        {"depth": 1, "aspect_ratio": 64, "head_dim": 128, "N_scaling": 100},
        {"depth": 2, "aspect_ratio": 64, "head_dim": 128, "N_scaling": 200},
        {"depth": 4, "aspect_ratio": 64, "head_dim": 128, "N_scaling": 400},
        {"depth": 2, "aspect_ratio": 72, "head_dim": 128, "N_scaling": 9_961_496},
        {"depth": 4, "aspect_ratio": 72, "head_dim": 128, "N_scaling": 19_660_872},
        {"depth": 7, "aspect_ratio": 72, "head_dim": 128, "N_scaling": 38_797_504},
    ]

    ranked = rank_geometric_triplets(rows, exact_only=True, max_control_changes=0)

    assert len(ranked) == 1
    assert ranked[0]["is_exact_geometric"] is True
    assert ranked[0]["n_values"] == "100,200,400"


def test_rank_geometric_triplets_exact_only_rejects_near_misses() -> None:
    rows = [
        {"depth": 2, "aspect_ratio": 72, "head_dim": 128, "N_scaling": 9_961_496},
        {"depth": 4, "aspect_ratio": 72, "head_dim": 128, "N_scaling": 19_660_872},
        {"depth": 7, "aspect_ratio": 72, "head_dim": 128, "N_scaling": 38_797_504},
    ]

    with pytest.raises(ValueError, match="no exact geometric size triplet"):
        rank_geometric_triplets(rows, exact_only=True)


def test_rank_geometric_triplets_can_restrict_to_fixed_controls() -> None:
    rows = [
        {"depth": 2, "aspect_ratio": 48, "head_dim": 128, "N_scaling": 100.0},
        {"depth": 3, "aspect_ratio": 48, "head_dim": 128, "N_scaling": 230.0},
        {"depth": 7, "aspect_ratio": 48, "head_dim": 128, "N_scaling": 520.0},
        {"depth": 2, "aspect_ratio": 64, "head_dim": 128, "N_scaling": 100.0},
        {"depth": 5, "aspect_ratio": 80, "head_dim": 128, "N_scaling": 220.0},
        {"depth": 8, "aspect_ratio": 96, "head_dim": 128, "N_scaling": 484.0},
    ]

    ranked = rank_geometric_triplets(rows, max_control_changes=0)

    assert ranked
    assert {row["control_changes"] for row in ranked} == {0}
    assert ranked[0]["aspect_ratios"] == "48,48,48"


def test_select_geometric_triplet_obeys_max_value_bound() -> None:
    rows = [
        {"depth": 1, "aspect_ratio": 64, "N_scaling": 100.0},
        {"depth": 2, "aspect_ratio": 64, "N_scaling": 200.0},
        {"depth": 4, "aspect_ratio": 64, "N_scaling": 400.0},
        {"depth": 8, "aspect_ratio": 64, "N_scaling": 800.0},
    ]

    selected = select_geometric_triplet(rows, max_value=500.0)

    assert [row["N_scaling"] for row in selected] == [100.0, 200.0, 400.0]


def test_select_geometric_triplet_rejects_degenerate_equal_size_triplet() -> None:
    rows = [
        {"depth": 1, "aspect_ratio": 48, "N_scaling": 100.0},
        {"depth": 2, "aspect_ratio": 48, "N_scaling": 100.0},
        {"depth": 3, "aspect_ratio": 48, "N_scaling": 100.0},
        {"depth": 2, "aspect_ratio": 96, "N_scaling": 200.0},
        {"depth": 4, "aspect_ratio": 96, "N_scaling": 410.0},
        {"depth": 8, "aspect_ratio": 96, "N_scaling": 850.0},
    ]

    selected = select_geometric_triplet(rows)

    assert [row["N_scaling"] for row in selected] == [200.0, 410.0, 850.0]


def test_run_train_array_accepts_architecture_columns_from_plan() -> None:
    script = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "csc_fast_scaling_law"
        / "run_train_array.sbatch"
    ).read_text(encoding="utf-8")

    assert "ASPECT_RATIO=\"${ASPECT_RATIO:-64}\"" in script
    assert "HEAD_DIM=\"${HEAD_DIM:-128}\"" in script
    assert "--aspect-ratio=\"$ASPECT_RATIO\"" in script
    assert "--head-dim=\"$HEAD_DIM\"" in script
