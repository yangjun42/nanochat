from __future__ import annotations

import inspect

import pytest
import torch

from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit
from scripts.base_eval import (
    make_bpb_eval_plan,
    make_manifest_bpb_eval_plan,
    parse_bpb_val_row_group_starts,
    validate_runtime_contract,
)


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


def test_manifest_bpb_plan_uses_independent_heldout_names() -> None:
    plan = make_manifest_bpb_eval_plan("train,val", ["heldout-a", "heldout-b"])

    assert plan == [
        ("train", None, "train"),
        ("val", "heldout-a", "val/heldout-a"),
        ("val", "heldout-b", "val/heldout-b"),
    ]


def test_eval_runtime_contract_records_exact_backend_and_dtype() -> None:
    assert validate_runtime_contract(
        required_attention_backend="fa3",
        required_compute_dtype="bfloat16",
        use_fa3=True,
        compute_dtype=torch.bfloat16,
    ) == {"attention_backend": "fa3", "compute_dtype": "bfloat16"}


@pytest.mark.parametrize(
    ("required_backend", "required_dtype", "use_fa3", "compute_dtype"),
    [
        ("fa3", "bfloat16", False, torch.bfloat16),
        ("fa3", "bfloat16", True, torch.float32),
    ],
)
def test_eval_runtime_contract_fails_closed_on_mismatch(
    required_backend: str,
    required_dtype: str,
    use_fa3: bool,
    compute_dtype: torch.dtype,
) -> None:
    with pytest.raises(RuntimeError, match="evaluation .* contract violated"):
        validate_runtime_contract(
            required_attention_backend=required_backend,
            required_compute_dtype=required_dtype,
            use_fa3=use_fa3,
            compute_dtype=compute_dtype,
        )
