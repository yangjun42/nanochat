# 1/2/4-GPU numerical preflight

This is a diagnostic, not a training campaign. Its manifest always records
`formal_campaign_gpu_count=4`; measuring 1 and 2 GPUs cannot alter the formal
campaign configuration.

The important control is the input: every world size reads the same hashed
`global_token_batches.pt`. The runner never calls the parquet/manifest data
loader. It only partitions each already-materialized global batch into equal,
contiguous rank shares. Model configuration, initialization seed, optimizer,
schedule, microbatch size, runtime backend, and dtype are frozen once in
`preflight_manifest.json`.

First capture the formal environment lock inside a four-GPU `gputest`
allocation, after sourcing the exact runtime bootstrap, setting the pinned
`NANOCHAT_FA3_REVISION`, and making the kernel cache immutable/offline:

```bash
python -m csc_fast_scaling_law.environment_probe \
  --python /path/to/formal/python \
  --checkout /path/to/clean/nanochat \
  --lockfile /path/to/nanochat/uv.lock \
  --out /path/kept-private/environment_lock.json
```

The environment lock must exist before preparing this preflight. The preflight
copies it and the runtime bootstrap into its private input directory and binds
their SHA-256 hashes; source paths are not written to its manifest. The
orchestrator and every 1/2/4-process worker rerun the same formal environment
probe/validator. This checks the Python and dependency versions, CUDA/NCCL,
dtype/backend, requested and resolved FA3 revision, interface origin, and all
loaded FA3 file hashes.

Also create a private formal runtime profile. The current intended profile is:

```json
{
  "schema_version": "nanochat_formal_runtime_profile/v1",
  "gpu_count": 4,
  "compute_dtype": "bfloat16",
  "attention_backend": "fa3",
  "compile_model": true,
  "fp8_recipe": null
}
```

`compile_model` is fixed to `true`, matching `base_train`. If the formal recipe
later changes, including enabling tensorwise FP8, create a new profile and a
new preflight manifest; an old result cannot validate the changed runtime.

Prepare the immutable inputs from a clean checkout. The defaults use eight
updates, a global batch of 256 sequences at length 2048 (524,288 prediction
positions per update), and the small depth-4/width-128 diagnostic model:

```bash
python -m csc_fast_scaling_law.gpu_preflight prepare \
  --output-dir /path/kept-private/preflight-input \
  --environment-lock /path/kept-private/environment_lock.json \
  --runtime-bootstrap /path/kept-private/runtime_bootstrap.sh \
  --formal-runtime-profile /path/kept-private/formal_runtime_profile.json
```

To use externally packed tokens, save an integer tensor shaped
`[num_updates, global_batch_sequences, sequence_length + 1]`, then pass it with
`--input-token-tensor`. The source path is not copied into the manifest. Only
the copied tensor, its content hash, dimensions, and the generic source label
are recorded.

Submit one four-GPU node. Account and partition remain command-line choices so
the tracked script contains no site or user identity:

```bash
export NANOCHAT_CHECKOUT=/path/to/clean/nanochat
export PREFLIGHT_MANIFEST=/path/kept-private/preflight-input/preflight_manifest.json
export PREFLIGHT_OUTPUT_ROOT=/path/kept-private/preflight-results
export NANOCHAT_PYTHON=/path/to/python
export RUNTIME_BOOTSTRAP=/path/kept-private/runtime_bootstrap.sh
sbatch --account=YOUR_ACCOUNT --partition=gputest \
  csc_fast_scaling_law/run_gpu_preflight.sbatch
```

The wrapper reads the expected bootstrap hash from the manifest and verifies it
before sourcing. It exports that hash to the orchestrator and workers, which
verify both the sourced file and the copied evidence file again.

The job runs three fresh `torchrun` invocations in order: 1, 2, then 4 GPUs.
Each one writes its losses, exact initial/final parameter states, parameter
update norm, timing samples, and provenance. `summary.json` has two deliberately
separate sections:

- `numerical_equivalence` is the only gate. Version 1 is exact-only: all four
  tolerances are frozen at zero. It compares every recorded loss and the full
  final parameter tensors against the one-GPU reference. Any nonzero difference
  produces `FAIL` while preserving its measured magnitude for human review.
- `throughput` is `informational_only`. It reports speedup and parallel
  efficiency but never changes PASS/FAIL.

Do not inspect a failure, tune a threshold, and rerun it as if the new PASS had
been preregistered. A nonzero tolerance would require a separately designed,
human-approved calibration protocol and a new schema; this v1 implementation
does not accept one. A failure also must not be waived because throughput looks
good.
