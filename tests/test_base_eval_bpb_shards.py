from __future__ import annotations

import inspect

from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit
from scripts.base_eval import make_bpb_eval_plan, parse_bpb_val_row_group_starts


def test_parse_bpb_val_row_group_starts() -> None:
    assert parse_bpb_val_row_group_starts("0,16,32") == [0, 16, 32]
    assert parse_bpb_val_row_group_starts(" 0 ") == [0]


def test_make_bpb_eval_plan_preserves_default_val_key() -> None:
    plan = make_bpb_eval_plan("train,val", [0])

    assert plan == [
        ("train", 0, "train"),
        ("val", 0, "val"),
    ]


def test_make_bpb_eval_plan_names_multiple_validation_shards() -> None:
    plan = make_bpb_eval_plan("val", [0, 16])

    assert plan == [
        ("val", 0, "val@rg0"),
        ("val", 16, "val@rg16"),
    ]


def test_bos_bestfit_loader_exposes_row_group_start() -> None:
    signature = inspect.signature(tokenizing_distributed_data_loader_with_state_bos_bestfit)

    assert "row_group_start" in signature.parameters
