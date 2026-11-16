"""
Distributed dataloaders for pretraining.

BOS-aligned bestfit:
   - Every row starts with BOS token
   - Documents packed using best-fit algorithm to minimize cropping
   - When no document fits remaining space, crops a document to fill exactly
   - 100% utilization (no padding), ~35% tokens cropped at T=2048

Compared to the original tokenizing_distributed_data_loader:
BOS-aligned loses ~35% of tokens to cropping, but ensures that
there are fewer "confusing" tokens in the train/val batches as every token can
now attend back to the BOS token and sees the full context of the document.

Fallback to the original if you have very limited data AND long documents:
https://github.com/karpathy/nanochat/blob/3c3a3d7/nanochat/dataloader.py#L78-L117
"""

import hashlib
import os
import warnings
from functools import lru_cache
from pathlib import Path

import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files, load_parquet_manifest


LOADER_STATE_VERSION = 2


@lru_cache(maxsize=16)
def _load_manifest_cached(
    manifest_path: str,
    expected_split: str,
    manifest_mtime_ns: int,
    manifest_size: int,
):
    # mtime/size are deliberately part of the cache key. Their values are not
    # otherwise needed, but prevent reusing a parsed manifest after an edit.
    del manifest_mtime_ns, manifest_size
    return load_parquet_manifest(manifest_path, expected_split=expected_split)


def _load_manifest_for_stream(manifest_path: str, expected_split: str):
    resolved = Path(manifest_path).expanduser().resolve()
    stat = os.stat(resolved)
    return _load_manifest_cached(
        str(resolved), expected_split, stat.st_mtime_ns, stat.st_size
    )


def _deterministic_row_group_order(
    parquet_paths: list[str],
    *,
    data_seed: int,
    epoch: int,
    namespace: str,
) -> list[tuple[int, int]]:
    """Return a stable global permutation of every row group in every shard.

    Hash sorting avoids relying on a library-specific PRNG implementation. The
    permutation is global first and only then divided between DDP ranks.
    """

    row_group_counts = [pq.ParquetFile(filepath).num_row_groups for filepath in parquet_paths]
    return _deterministic_row_group_order_from_counts(
        row_group_counts,
        data_seed=data_seed,
        epoch=epoch,
        namespace=namespace,
    )


def _deterministic_row_group_order_from_counts(
    row_group_counts: list[int],
    *,
    data_seed: int,
    epoch: int,
    namespace: str,
) -> list[tuple[int, int]]:
    catalog: list[tuple[int, int]] = [
        (pq_idx, rg_idx)
        for pq_idx, count in enumerate(row_group_counts)
        for rg_idx in range(count)
    ]

    def order_key(item: tuple[int, int]) -> bytes:
        pq_idx, rg_idx = item
        material = f"{namespace}\0{int(data_seed)}\0{int(epoch)}\0{pq_idx}\0{rg_idx}"
        return hashlib.sha256(material.encode("utf-8")).digest()

    return sorted(catalog, key=order_key)


def _manifest_source(
    split: str,
    *,
    dataset_manifest: str | None,
    validation_manifest: str | None,
    validation_shard: str | None,
):
    if split == "train":
        if dataset_manifest is None:
            raise ValueError("manifest-backed training requires dataset_manifest")
        manifest = _load_manifest_for_stream(dataset_manifest, expected_split="train")
        if validation_shard is not None:
            raise ValueError("validation_shard is only valid for split='val'")
        shards = list(manifest.shards)
    else:
        if validation_manifest is None:
            raise ValueError("manifest-backed validation requires validation_manifest")
        manifest = _load_manifest_for_stream(
            validation_manifest, expected_split="validation"
        )
        shards = list(manifest.shards)
        if validation_shard is not None:
            shards = [shard for shard in shards if shard.name == validation_shard]
            if not shards:
                available = ", ".join(shard.name for shard in manifest.shards)
                raise ValueError(
                    f"validation shard {validation_shard!r} is not in the manifest; "
                    f"available: {available}"
                )
    return manifest, [shard.path for shard in shards]


def _document_batches(
    split,
    resume_state_dict,
    tokenizer_batch_size,
    row_group_start=0,
    *,
    dataset_manifest=None,
    validation_manifest=None,
    data_seed=None,
    validation_shard=None,
    allow_data_reuse=False,
):
    """
    Infinite iterator over document batches (list of text strings) from parquet files.

    Handles DDP sharding and approximate resume. Each yield is (text_batch, (pq_idx, rg_idx, epoch))
    where text_batch is a list of document strings, indices track position for resumption,
    and epoch counts how many times we've cycled through the dataset (starts at 1).
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    using_manifest = dataset_manifest is not None or validation_manifest is not None
    if using_manifest:
        if row_group_start != 0:
            raise ValueError(
                "row_group_start is a legacy cursor and cannot be used with manifests; "
                "use data_seed for deterministic global shuffling"
            )
        if split == "train" and data_seed is None:
            raise ValueError("manifest-backed training requires an explicit data_seed")
        if resume_state_dict is not None:
            raise ValueError(
                "manifest resume is handled by the versioned output-batch cursor, "
                "not by parquet row-group position"
            )
        manifest, parquet_paths = _manifest_source(
            split,
            dataset_manifest=dataset_manifest,
            validation_manifest=validation_manifest,
            validation_shard=validation_shard,
        )
        assert parquet_paths, "manifest contains no selected parquet shards"
        parquet_files = [pq.ParquetFile(filepath) for filepath in parquet_paths]
        row_group_counts = [parquet_file.num_row_groups for parquet_file in parquet_files]
        epoch = 1
        while True:
            if split == "train":
                global_order = _deterministic_row_group_order_from_counts(
                    row_group_counts,
                    data_seed=int(data_seed),
                    epoch=epoch,
                    namespace=manifest.manifest_sha256,
                )
            else:
                # Heldout shards remain in declared order. They are independent
                # files, not nested offsets into one validation file.
                global_order = [
                    (pq_idx, rg_idx)
                    for pq_idx, count in enumerate(row_group_counts)
                    for rg_idx in range(count)
                ]
            local_order = global_order[ddp_rank::ddp_world_size]
            if not local_order:
                raise ValueError(
                    "selected data has fewer row groups than DDP ranks; "
                    "use fewer ranks or more row groups"
                )
            for pq_idx, rg_idx in local_order:
                rg = parquet_files[pq_idx].read_row_group(rg_idx)
                batch = rg.column("text").to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i + tokenizer_batch_size], (pq_idx, rg_idx, epoch)
            if not allow_data_reuse:
                raise RuntimeError(
                    "manifest-backed data was exhausted after one epoch; refusing "
                    "to reuse examples within a run. Add more shards or explicitly "
                    "set allow_data_reuse=True for a non-primary diagnostic run."
                )
            epoch += 1
        return

    warn_on_legacy = ddp_rank == 0 and split == "train" # rank 0 on train split will warn on legacy
    if warn_on_legacy:
        warnings.warn(
            "Using the legacy implicit train/validation split. New experiments "
            "must pass dataset_manifest, validation_manifest, and data_seed.",
            FutureWarning,
            stacklevel=2,
        )
    if row_group_start != 0:
        warnings.warn(
            "row_group_start is a deprecated legacy cursor, not a random seed; "
            "it drops a prefix from every parquet shard. Use a manifest and data_seed.",
            FutureWarning,
            stacklevel=2,
        )
    parquet_paths = list_parquet_files(warn_on_legacy=warn_on_legacy)
    assert len(parquet_paths) != 0, "No dataset parquet files found, did you run dataset.py?"
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
    row_group_start = int(row_group_start)
    first_pass = True
    pq_idx = resume_pq_idx
    epoch = resume_epoch

    while True:  # iterate infinitely (multi-epoch)
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            pf = pq.ParquetFile(filepath)
            # Start from resume point if resuming on same file, otherwise from DDP rank
            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                base_idx = resume_rg_idx // ddp_world_size
                base_idx += 1  # advance by 1 so we don't repeat data after resuming
                rg_idx = base_idx * ddp_world_size + ddp_rank
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None  # only do this once
            else:
                rg_idx = row_group_start + ddp_rank
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx, epoch)
                rg_idx += ddp_world_size
            pq_idx += 1
        first_pass = False
        epoch += 1


def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    buffer_size=1000,
    row_group_start=0,
    dataset_manifest=None,
    validation_manifest=None,
    data_seed=None,
    validation_shard=None,
    allow_data_reuse=False,
):
    """
    BOS-aligned dataloader with Best-Fit Cropping.

    Reduces token waste compared to simple greedy cropping by searching a buffer
    for documents that fit well, while maintaining 100% utilization (no padding).

    Algorithm for each row:
    1. From buffered docs, pick the LARGEST doc that fits entirely
    2. Repeat until no doc fits
    3. When nothing fits, crop a doc to fill remaining space exactly

    Key properties:
    - Every row starts with BOS
    - 100% utilization (no padding, every token is trained on)
    - Approximately 35% of all tokens are discarded due to cropping
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    if B <= 0 or T <= 0:
        raise ValueError("B and T must be positive")
    if tokenizer_batch_size <= 0 or buffer_size <= 0:
        raise ValueError("tokenizer_batch_size and buffer_size must be positive")

    using_manifest = dataset_manifest is not None or validation_manifest is not None
    if using_manifest and not (dataset_manifest and validation_manifest):
        raise ValueError(
            "dataset_manifest and validation_manifest must be supplied together"
        )
    if using_manifest and split == "train" and data_seed is None:
        raise ValueError("manifest-backed training requires an explicit data_seed")
    if using_manifest and row_group_start != 0:
        raise ValueError(
            "row_group_start cannot be combined with a manifest; use data_seed"
        )

    resume_batch_index = 0
    data_contract = None
    document_resume_state = resume_state_dict
    if using_manifest:
        train_manifest = _load_manifest_for_stream(dataset_manifest, expected_split="train")
        val_manifest = _load_manifest_for_stream(
            validation_manifest, expected_split="validation"
        )
        _, _, _, contract_world_size = get_dist_info()
        data_contract = {
            "loader_state_version": LOADER_STATE_VERSION,
            "train_manifest_sha256": train_manifest.manifest_sha256,
            "validation_manifest_sha256": val_manifest.manifest_sha256,
            "data_seed": int(data_seed) if data_seed is not None else None,
            "split": split,
            "validation_shard": validation_shard,
            "device_batch_size": int(B),
            "sequence_length": int(T),
            "tokenizer_batch_size": int(tokenizer_batch_size),
            "packing_buffer_size": int(buffer_size),
            "ddp_world_size": contract_world_size,
            "allow_data_reuse": bool(allow_data_reuse),
        }
        if resume_state_dict is not None:
            if resume_state_dict.get("loader_state_version") != LOADER_STATE_VERSION:
                raise ValueError(
                    "cannot exactly resume a manifest run from a legacy dataloader state"
                )
            prior_contract = resume_state_dict.get("data_contract")
            if prior_contract != data_contract:
                raise ValueError(
                    "resume data contract differs from the checkpoint; refusing to "
                    "change manifests, data_seed, split, or heldout shard"
                )
            resume_batch_index = int(resume_state_dict["batch_index"])
            if resume_batch_index < 0:
                raise ValueError("resume batch_index must be non-negative")
        # Exact recovery replays deterministic packing up to the currently
        # prefetched (unconsumed) output batch. No parquet cursor is advanced.
        document_resume_state = None

    row_capacity = T + 1
    batches = _document_batches(
        split,
        document_resume_state,
        tokenizer_batch_size,
        row_group_start=row_group_start,
        dataset_manifest=dataset_manifest,
        validation_manifest=validation_manifest,
        data_seed=data_seed,
        validation_shard=validation_shard,
        allow_data_reuse=allow_data_reuse,
    )
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    pq_idx, rg_idx, epoch = 0, 0, 1
    packing_stats = {
        "documents_consumed": 0,
        "documents_cropped": 0,
        "source_tokens_consumed": 0,
        "cropped_tokens": 0,
    }
    output_batch_index = 0

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch
        doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
        for tokens in token_lists:
            doc_buffer.append(tokens)

    # Pre-allocate buffers once: layout is [inputs (B*T) | targets (B*T)]
    # This gives us contiguous views and a single HtoD transfer
    device = torch.device(device)
    use_cuda = device.type == "cuda"
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long) # for building rows without creating Python lists
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda) # staging area (CPU)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device) # on-device buffer
    cpu_inputs = cpu_buffer[:B * T].view(B, T) # a few views into these buffers just for convenience
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)
    transfer_done = None

    while True:
        packing_before_batch = dict(packing_stats)
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                # Ensure buffer has documents
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    doc_len = len(doc)
                    row_buffer[row_idx, pos:pos + doc_len] = torch.tensor(doc, dtype=torch.long)
                    pos += doc_len
                    packing_stats["documents_consumed"] += 1
                    packing_stats["source_tokens_consumed"] += doc_len
                else:
                    # No doc fits - crop shortest in buffer to fill remaining and minimize waste
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    packing_stats["documents_consumed"] += 1
                    packing_stats["source_tokens_consumed"] += len(doc)
                    discarded = max(0, len(doc) - remaining)
                    if discarded:
                        packing_stats["documents_cropped"] += 1
                        packing_stats["cropped_tokens"] += discarded
                    pos += remaining

        # Copy to pinned CPU buffer, then single HtoD transfer
        # The previous asynchronous copy may still be reading this persistent
        # staging buffer. Packing into row_buffer can overlap it; overwriting
        # cpu_buffer must wait until the transfer itself has completed.
        if transfer_done is not None:
            transfer_done.synchronize()
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])

        if using_manifest:
            _, ddp_rank, _, ddp_world_size = get_dist_info()
            predictions_per_rank = B * T
            state_dict = {
                "loader_state_version": LOADER_STATE_VERSION,
                # This identifies the batch held in x/y. Training checkpoints
                # save the currently prefetched, still-unconsumed batch, so an
                # exact resume must yield this same batch again.
                "batch_index": output_batch_index,
                "next_batch_index": output_batch_index + 1,
                "pq_idx": pq_idx,
                "rg_idx": rg_idx,
                "epoch": epoch,
                "ddp_rank": ddp_rank,
                "prediction_positions_before_batch": (
                    output_batch_index * predictions_per_rank * ddp_world_size
                ),
                "prediction_positions_after_batch": (
                    (output_batch_index + 1) * predictions_per_rank * ddp_world_size
                ),
                "packing_before_batch": packing_before_batch,
                "packing_after_batch": dict(packing_stats),
                "data_contract": data_contract,
            }
        else:
            state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}

        if using_manifest and output_batch_index < resume_batch_index:
            output_batch_index += 1
            continue

        # Single HtoD copy into persistent GPU buffer and yield
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        if use_cuda:
            if transfer_done is None:
                transfer_done = torch.cuda.Event()
            transfer_done.record(torch.cuda.current_stream(device))
        yield inputs, targets, state_dict
        output_batch_index += 1

def tokenizing_distributed_data_loader_bos_bestfit(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state_bos_bestfit(*args, **kwargs):
        yield inputs, targets
