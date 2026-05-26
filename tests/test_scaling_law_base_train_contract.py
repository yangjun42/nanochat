from __future__ import annotations

from pathlib import Path


def test_base_train_exposes_scaling_law_campaign_arguments_and_logs() -> None:
    source = (Path(__file__).resolve().parents[1] / "scripts" / "base_train.py").read_text(encoding="utf-8")

    assert 'parser.add_argument("--seed"' in source
    assert 'parser.add_argument("--target-tokens"' in source
    assert "torch.manual_seed(args.seed)" in source
    assert "Number of scaling parameters" in source
    assert "Target training tokens" in source
    assert "Horizon source" in source
    assert "D_actual" in source
