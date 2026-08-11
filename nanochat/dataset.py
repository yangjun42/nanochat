"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- download the files on demand if they are not on disk

For details of how the dataset was prepared, see `repackage_data_reference.py`.
"""

import os
import argparse
import hashlib
import json
import time
import requests
import pyarrow as pa
import pyarrow.parquet as pq
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset

# The URL on the internet where the data is hosted and downloaded from on demand
BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542 # the last datashard is shard_06542.parquet
index_to_filename = lambda index: f"shard_{index:05d}.parquet" # format of the filenames
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data_climbmix")

DATASET_MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ParquetShard:
    """One named parquet shard declared by a dataset manifest."""

    name: str
    path: str
    sha256: str | None = None
    size_bytes: int | None = None


@dataclass(frozen=True)
class ParquetManifest:
    """Validated, path-resolved manifest plus its immutable fingerprint."""

    source_path: str
    manifest_sha256: str
    dataset_id: str
    split: str
    shards: tuple[ParquetShard, ...]

    def provenance(self) -> dict[str, Any]:
        return {
            "schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
            "dataset_id": self.dataset_id,
            "split": self.split,
            "manifest_path": self.source_path,
            "manifest_sha256": self.manifest_sha256,
            "shards": [
                {
                    "name": shard.name,
                    "path": shard.path,
                    "sha256": shard.sha256,
                    "size_bytes": shard.size_bytes,
                }
                for shard in self.shards
            ],
        }


def _sha256_file(path: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_parquet_text_column(
    parquet_file: pq.ParquetFile,
    path: str | os.PathLike[str],
) -> None:
    """Cheaply validate the scalar text contract, scanning only if stats are absent."""

    resolved = str(Path(path).resolve())
    schema = parquet_file.schema_arrow
    text_index = schema.get_field_index("text")
    if text_index < 0:
        raise ValueError(
            f"manifest parquet shard has no required 'text' column: {resolved}"
        )
    text_type = schema.field(text_index).type
    if not (pa.types.is_string(text_type) or pa.types.is_large_string(text_type)):
        raise ValueError(
            "manifest parquet shard 'text' column must be string or large_string, "
            f"found {text_type} in {resolved}"
        )
    parquet_text_index = next(
        (
            index
            for index in range(parquet_file.metadata.num_columns)
            if parquet_file.metadata.row_group(0).column(index).path_in_schema == "text"
        ),
        None,
    )
    if parquet_text_index is None:
        raise ValueError(f"cannot resolve parquet leaf column for text: {resolved}")
    for row_group_index in range(parquet_file.num_row_groups):
        column_metadata = parquet_file.metadata.row_group(row_group_index).column(
            parquet_text_index
        )
        statistics = column_metadata.statistics
        null_count = None if statistics is None else statistics.null_count
        if null_count is None:
            null_count = parquet_file.read_row_group(
                row_group_index, columns=["text"]
            ).column("text").null_count
        if null_count:
            raise ValueError(
                f"manifest parquet shard contains {null_count} null text values "
                f"in row group {row_group_index}: {resolved}"
            )


def load_parquet_manifest(
    manifest_path: str | os.PathLike[str],
    *,
    expected_split: str | None = None,
    verify_shard_hashes: bool = False,
    require_shard_hashes: bool = False,
) -> ParquetManifest:
    """Load and validate a public, generic parquet manifest contract.

    A manifest is a JSON object with ``schema_version``, ``dataset_id``,
    ``split`` (``train`` or ``validation``), and a non-empty ``shards`` list.
    Each shard has a unique ``name`` and a ``path`` relative to the manifest
    (absolute paths are also accepted). ``sha256`` and ``size_bytes`` are
    optional declarations; when present, sizes are always checked and hashes
    are checked when ``verify_shard_hashes`` is true.

    The raw manifest bytes are hashed. This deliberately makes provenance
    sensitive to every manifest edit, even if the parsed JSON is equivalent.
    """

    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"dataset manifest does not exist: {path}")
    raw = path.read_bytes()
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON dataset manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("dataset manifest root must be a JSON object")
    schema_version = payload.get("schema_version")
    if schema_version != DATASET_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported dataset manifest schema_version={schema_version!r}; "
            f"expected {DATASET_MANIFEST_SCHEMA_VERSION}"
        )
    dataset_id = payload.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("dataset manifest dataset_id must be a non-empty string")
    split = payload.get("split")
    if split not in {"train", "validation"}:
        raise ValueError("dataset manifest split must be 'train' or 'validation'")
    if expected_split is not None and split != expected_split:
        raise ValueError(
            f"dataset manifest split mismatch: expected {expected_split!r}, got {split!r}"
        )
    shard_payloads = payload.get("shards")
    if not isinstance(shard_payloads, list) or not shard_payloads:
        raise ValueError("dataset manifest shards must be a non-empty list")

    shards: list[ParquetShard] = []
    names: set[str] = set()
    resolved_paths: set[str] = set()
    declared_hashes: set[str] = set()
    for index, shard_payload in enumerate(shard_payloads):
        if not isinstance(shard_payload, dict):
            raise ValueError(f"manifest shard {index} must be a JSON object")
        raw_path = shard_payload.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"manifest shard {index} path must be a non-empty string")
        shard_path = Path(raw_path).expanduser()
        if not shard_path.is_absolute():
            shard_path = path.parent / shard_path
        shard_path = shard_path.resolve()
        if shard_path.suffix != ".parquet":
            raise ValueError(f"manifest shard is not a parquet file: {shard_path}")
        if not shard_path.is_file():
            raise FileNotFoundError(f"manifest parquet shard does not exist: {shard_path}")
        resolved_path = str(shard_path)
        if resolved_path in resolved_paths:
            raise ValueError(f"manifest contains duplicate shard path: {resolved_path}")
        resolved_paths.add(resolved_path)

        name = shard_payload.get("name", shard_path.stem)
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"manifest shard {index} name must be a non-empty string")
        if name in names:
            raise ValueError(f"manifest contains duplicate shard name: {name}")
        names.add(name)

        declared_sha256 = shard_payload.get("sha256")
        if require_shard_hashes and declared_sha256 is None:
            raise ValueError(
                f"manifest shard {name!r} must declare sha256 for a contracted run"
            )
        if declared_sha256 is not None:
            if (
                not isinstance(declared_sha256, str)
                or len(declared_sha256) != 64
                or any(char not in "0123456789abcdefABCDEF" for char in declared_sha256)
            ):
                raise ValueError(f"manifest shard {name!r} has an invalid sha256")
            declared_sha256 = declared_sha256.lower()
            if declared_sha256 in declared_hashes:
                raise ValueError(
                    f"manifest contains duplicate shard content sha256: {declared_sha256}"
                )
            declared_hashes.add(declared_sha256)
        declared_size = shard_payload.get("size_bytes")
        if declared_size is not None:
            if not isinstance(declared_size, int) or declared_size < 0:
                raise ValueError(f"manifest shard {name!r} has an invalid size_bytes")
            actual_size = shard_path.stat().st_size
            if actual_size != declared_size:
                raise ValueError(
                    f"manifest shard {name!r} size mismatch: "
                    f"declared {declared_size}, found {actual_size}"
                )
        if verify_shard_hashes:
            if declared_sha256 is None:
                raise ValueError(
                    f"manifest shard {name!r} has no sha256 but hash verification was requested"
                )
            actual_sha256 = _sha256_file(resolved_path)
            if actual_sha256 != declared_sha256:
                raise ValueError(
                    f"manifest shard {name!r} sha256 mismatch: "
                    f"declared {declared_sha256}, found {actual_sha256}"
                )
        # Opening metadata catches truncated/corrupt parquet footers cheaply.
        parquet_file = pq.ParquetFile(resolved_path)
        if parquet_file.num_row_groups <= 0 or parquet_file.metadata.num_rows <= 0:
            raise ValueError(f"manifest parquet shard is empty: {resolved_path}")
        validate_parquet_text_column(parquet_file, resolved_path)
        shards.append(
            ParquetShard(
                name=name,
                path=resolved_path,
                sha256=declared_sha256,
                size_bytes=declared_size,
            )
        )

    return ParquetManifest(
        source_path=str(path),
        manifest_sha256=manifest_sha256,
        dataset_id=dataset_id,
        split=split,
        shards=tuple(shards),
    )


def validation_shard_names(validation_manifest: str | os.PathLike[str]) -> list[str]:
    """Return heldout shard names in the manifest's preregistered order."""

    manifest = load_parquet_manifest(validation_manifest, expected_split="validation")
    return [shard.name for shard in manifest.shards]


def validate_manifest_separation(
    train_manifest: ParquetManifest,
    validation_manifest: ParquetManifest,
) -> None:
    """Fail closed on exact train/heldout overlap visible in the manifests."""

    if train_manifest.split != "train" or validation_manifest.split != "validation":
        raise ValueError("manifest separation requires train and validation manifests")
    if train_manifest.dataset_id != validation_manifest.dataset_id:
        raise ValueError(
            "training and validation manifests declare different dataset_id values: "
            f"{train_manifest.dataset_id!r} != {validation_manifest.dataset_id!r}"
        )
    train_paths = {shard.path for shard in train_manifest.shards}
    validation_paths = {shard.path for shard in validation_manifest.shards}
    overlapping_paths = sorted(train_paths & validation_paths)
    if overlapping_paths:
        raise ValueError(
            "training and validation manifests reference the same parquet path: "
            + ", ".join(overlapping_paths)
        )
    train_hashes = {shard.sha256 for shard in train_manifest.shards if shard.sha256}
    validation_hashes = {
        shard.sha256 for shard in validation_manifest.shards if shard.sha256
    }
    overlapping_hashes = sorted(train_hashes & validation_hashes)
    if overlapping_hashes:
        raise ValueError(
            "training and validation manifests contain exact duplicate shard content: "
            + ", ".join(overlapping_hashes)
        )

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported

def list_parquet_files(data_dir=None, warn_on_legacy=False):
    """ Looks into a data dir and returns full paths to all parquet files. """
    data_dir = DATA_DIR if data_dir is None else data_dir

    # Legacy-supporting code due to the upgrade from FinewebEdu-100B to ClimbMix-400B
    # This code will eventually be deleted.
    if not os.path.exists(data_dir):
        if warn_on_legacy:
            print()
            print("=" * 80)
            print("  WARNING: DATASET UPGRADE REQUIRED")
            print("=" * 80)
            print()
            print(f"  Could not find: {data_dir}")
            print()
            print("  nanochat recently switched from FinewebEdu-100B to ClimbMix-400B.")
            print("  Everyone who does `git pull` as of March 4, 2026 is expected to see this message.")
            print("  To upgrade to the new ClimbMix-400B dataset, run these two commands:")
            print()
            print("    python -m nanochat.dataset -n 170     # download ~170 shards, enough for GPT-2, adjust as desired")
            print("    python -m scripts.tok_train           # re-train tokenizer on new ClimbMix data")
            print()
            print("  For now, falling back to your old FinewebEdu-100B dataset...")
            print("=" * 80)
            print()
        # attempt a fallback to the legacy data directory
        data_dir = os.path.join(base_dir, "base_data")

    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths

def parquets_iter_batched(split, start=0, step=1):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts

# -----------------------------------------------------------------------------
def download_single_file(index):
    """ Downloads a single file index, with some backoff """

    # Construct the local filepath for this file and skip if it already exists
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    # Construct the remote URL for this file
    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    # Download with retries
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            # Write to temporary file first
            temp_path = filepath + f".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
            # Move temp file to final location
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            # Clean up any partial files
            for path in [filepath + f".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            # Try a few times with exponential backoff: 2^attempt seconds
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pretraining dataset shards")
    parser.add_argument("-n", "--num-files", type=int, default=-1, help="Number of train shards to download (default: -1), -1 = disable")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers (default: 4)")
    args = parser.parse_args()

    # Prepare the output directory
    os.makedirs(DATA_DIR, exist_ok=True)

    # The way this works is that the user specifies the number of train shards to download via the -n flag.
    # In addition to that, the validation shard is *always* downloaded and is pinned to be the last shard.
    num_train_shards = MAX_SHARD if args.num_files == -1 else min(args.num_files, MAX_SHARD)
    ids_to_download = list(range(num_train_shards))
    ids_to_download.append(MAX_SHARD) # always download the validation shard

    # Download the shards
    print(f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers...")
    print(f"Target directory: {DATA_DIR}")
    print()
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_single_file, ids_to_download)

    # Report results
    successful = sum(1 for success in results if success)
    print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {DATA_DIR}")
