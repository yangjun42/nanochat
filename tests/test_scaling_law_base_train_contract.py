from __future__ import annotations

from pathlib import Path


def test_base_train_exposes_scaling_law_campaign_arguments_and_logs() -> None:
    source = (Path(__file__).resolve().parents[1] / "scripts" / "base_train.py").read_text(encoding="utf-8")

    assert 'parser.add_argument("--seed"' in source
    assert 'parser.add_argument("--target-tokens"' in source
    assert 'parser.add_argument("--muon-momentum-warmup-steps"' in source
    assert 'parser.add_argument("--weight-decay-horizon-tokens"' in source
    assert "torch.manual_seed(args.seed)" in source
    assert "Number of scaling parameters" in source
    assert "Target training tokens" in source
    assert "Horizon source" in source
    assert "D_actual" in source
    assert "weight_decay_horizon_tokens" in source
    assert "current_muon_momentum = get_muon_momentum(step)" in source
    assert "\n    muon_momentum = get_muon_momentum(step)" not in source


def test_roihu_train_harness_forwards_optimizer_horizon_controls() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "csc_fast_scaling_law"
        / "run_train_array.sbatch"
    ).read_text(encoding="utf-8")

    assert '--muon-momentum-warmup-steps="${MUON_MOMENTUM_WARMUP_STEPS:-400}"' in source
    assert '--weight-decay-horizon-tokens="${WEIGHT_DECAY_HORIZON_TOKENS:--1}"' in source
