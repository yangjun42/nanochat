from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import nanochat.dataloader as dataloader_module

from nanochat.dataloader import (
    _deterministic_row_group_order,
    tokenizing_distributed_data_loader_with_state_bos_bestfit,
)


class IntegerTokenizer:
    def get_bos_token_id(self) -> int:
        return 99

    def encode(self, batch, prepend, num_threads):
        del num_threads
        return [[prepend, int(text)] for text in batch]


@pytest.mark.parametrize("device", ["cuda:0", torch.device("cuda", 0)])
def test_cuda_loader_waits_before_reusing_pinned_staging(monkeypatch, device):
    events = []
    real_empty = torch.empty
    real_copy = torch.Tensor.copy_

    def empty(*args, **kwargs):
        if "pin_memory" in kwargs:
            events.append(("pin_memory", kwargs.pop("pin_memory")))
        if "device" in kwargs:
            kwargs["device"] = "cpu"
        return real_empty(*args, **kwargs)

    def copy(tensor, source, non_blocking=False):
        events.append(("copy", non_blocking))
        return real_copy(tensor, source, non_blocking=non_blocking)

    class Event:
        def record(self, stream):
            events.append(("record", stream))

        def synchronize(self):
            events.append(("wait", None))

    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(torch.Tensor, "copy_", copy)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: "copy-stream")
    monkeypatch.setattr(
        dataloader_module,
        "_document_batches",
        lambda *args, **kwargs: iter([(["1"], (0, 0, 1)), (["2"], (0, 1, 1))]),
    )
    loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        IntegerTokenizer(), B=1, T=1, split="train", device=device, buffer_size=1,
    )
    _, first_targets, _ = next(loader)
    assert first_targets.item() == 1
    _, second_targets, _ = next(loader)
    assert second_targets.item() == 2
    assert events == [
        ("pin_memory", True),
        ("copy", False), ("copy", False), ("copy", True), ("record", "copy-stream"),
        ("wait", None),
        ("copy", False), ("copy", False), ("copy", True), ("record", "copy-stream"),
    ]


def _write_parquet(path: Path, values: list[str]) -> None:
    pq.write_table(pa.table({"text": values}), path, row_group_size=1)


def _write_manifest(path: Path, split: str, shard_paths: list[Path]) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_id": "unit-test",
                "split": split,
                "shards": [
                    {"name": shard.stem, "path": shard.name}
                    for shard in shard_paths
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _contract_files(tmp_path: Path) -> tuple[Path, Path, list[Path]]:
    train_a = tmp_path / "train-a.parquet"
    train_b = tmp_path / "train-b.parquet"
    heldout_a = tmp_path / "heldout-a.parquet"
    heldout_b = tmp_path / "heldout-b.parquet"
    _write_parquet(train_a, ["1", "2", "3"])
    _write_parquet(train_b, ["4", "5", "6"])
    _write_parquet(heldout_a, ["7", "8"])
    _write_parquet(heldout_b, ["9", "10"])
    train_manifest = _write_manifest(tmp_path / "train.json", "train", [train_a, train_b])
    val_manifest = _write_manifest(
        tmp_path / "validation.json",
        "validation",
        [heldout_a, heldout_b],
    )
    return train_manifest, val_manifest, [train_a, train_b]


def test_global_row_group_order_is_seeded_and_reproducible(tmp_path: Path) -> None:
    _, _, train_paths = _contract_files(tmp_path)

    order_a = _deterministic_row_group_order(
        [str(path) for path in train_paths], data_seed=7, epoch=1, namespace="manifest"
    )
    order_b = _deterministic_row_group_order(
        [str(path) for path in train_paths], data_seed=7, epoch=1, namespace="manifest"
    )
    order_c = _deterministic_row_group_order(
        [str(path) for path in train_paths], data_seed=8, epoch=1, namespace="manifest"
    )

    assert order_a == order_b
    assert order_a != order_c
    assert sorted(order_a) == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]


def test_manifest_loader_resume_replays_current_prefetched_batch(tmp_path: Path) -> None:
    train_manifest, val_manifest, _ = _contract_files(tmp_path)
    kwargs = {
        "tokenizer": IntegerTokenizer(),
        "B": 1,
        "T": 1,
        "split": "train",
        "device": "cpu",
        "buffer_size": 1,
        "tokenizer_batch_size": 1,
        "dataset_manifest": str(train_manifest),
        "validation_manifest": str(val_manifest),
        "data_seed": 123,
    }
    loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(**kwargs)
    first_x, first_y, first_state = next(loader)
    first_x, first_y = first_x.clone(), first_y.clone()
    second_x, second_y, second_state = next(loader)
    second_x, second_y = second_x.clone(), second_y.clone()

    resumed = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        **kwargs,
        resume_state_dict=second_state,
    )
    resumed_x, resumed_y, resumed_state = next(resumed)

    assert first_state["batch_index"] == 0
    assert second_state["batch_index"] == 1
    assert torch.equal(resumed_x, second_x)
    assert torch.equal(resumed_y, second_y)
    assert resumed_state == second_state
    assert not torch.equal(first_y, second_y)


def test_manifest_validation_selects_independent_heldout_file(tmp_path: Path) -> None:
    train_manifest, val_manifest, _ = _contract_files(tmp_path)
    loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        IntegerTokenizer(),
        B=1,
        T=1,
        split="val",
        device="cpu",
        buffer_size=1,
        tokenizer_batch_size=1,
        dataset_manifest=str(train_manifest),
        validation_manifest=str(val_manifest),
        data_seed=123,
        validation_shard="heldout-b",
    )

    _, targets, state = next(loader)

    assert targets.item() in {9, 10}
    assert state["data_contract"]["validation_shard"] == "heldout-b"


def test_manifest_loader_rejects_legacy_row_group_cursor(tmp_path: Path) -> None:
    train_manifest, val_manifest, _ = _contract_files(tmp_path)
    loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        IntegerTokenizer(),
        B=1,
        T=1,
        split="train",
        device="cpu",
        dataset_manifest=str(train_manifest),
        validation_manifest=str(val_manifest),
        data_seed=1,
        row_group_start=1,
    )

    with pytest.raises(ValueError, match="row_group_start"):
        next(loader)


def test_manifest_loader_refuses_a_second_epoch_by_default(tmp_path: Path) -> None:
    train_manifest, val_manifest, _ = _contract_files(tmp_path)
    kwargs = {
        "tokenizer": IntegerTokenizer(),
        "B": 1,
        "T": 1,
        "split": "train",
        "device": "cpu",
        "buffer_size": 1,
        "tokenizer_batch_size": 1,
        "dataset_manifest": str(train_manifest),
        "validation_manifest": str(val_manifest),
        "data_seed": 123,
    }
    loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(**kwargs)
    for _ in range(6):
        next(loader)
    with pytest.raises(RuntimeError, match="refusing to reuse examples"):
        next(loader)

    diagnostic_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        **kwargs,
        allow_data_reuse=True,
    )
    batches = [next(diagnostic_loader) for _ in range(7)]
    assert batches[-1][2]["epoch"] == 2
    assert batches[-1][2]["data_contract"]["allow_data_reuse"] is True
