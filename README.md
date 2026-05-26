# Anonymous nanochat scaling-law harness

This branch contains only the generic runnable pieces needed to reproduce the
EMNLP 2026 nanochat model-size harness:

- `nanochat/`: core model, tokenizer, dataloader, checkpoint, evaluation, and
  training support code.
- `scripts/base_train.py`: base-model training entrypoint with `--depth`,
  `--aspect-ratio`, `--head-dim`, optional `--model-dim`, and token/FLOP horizon
  controls.
- `scripts/base_eval.py`: base-model evaluation entrypoint.
- `scaling_law_harness/model_shape.py`: shared architecture sizing helper.
- `scaling_law_harness/model_size_probe.py`: CPU-friendly parameter-count and
  `N_scaling` probe.

Cluster launchers, scheduler scripts, host-specific paths, run plans, logs, and
raw result files are intentionally excluded.

## Install

Use Python 3.10+ and install the CPU extra if you only need sizing probes:

```bash
uv sync --extra cpu --group dev
```

## Reproduce Parameter Counts

Formula-only counts do not allocate model weights:

```bash
uv run python -m scaling_law_harness.model_size_probe \
  --depths 2,4,7 \
  --aspect-ratios 72 \
  --head-dim 128 \
  --window-pattern L \
  --formula-counts \
  --out /tmp/nanochat_sizes.csv \
  --triplet-out /tmp/nanochat_triplets.csv \
  --print-triplet
```

The probe writes `N_total`, `N_scaling`, the estimated architecture FLOPs per
token, embedding/unembedding counts, transformer-matrix counts, and the resolved
`model_dim`/`n_head`. Omit `--formula-counts` to instantiate the model on the
PyTorch `meta` device and count parameters from nanochat modules directly.

To hold depth fixed and choose explicit widths nearest target `N_scaling`
values:

```bash
uv run python -m scaling_law_harness.model_size_probe \
  --depths 4 \
  --aspect-ratios 72 \
  --head-dim 128 \
  --target-n-scaling 25000000,50000000,100000000 \
  --formula-counts \
  --out /tmp/nanochat_width_sweep.csv
```

## Measure `N_scaling` During Training

`scripts/base_train.py` prints the parameter breakdown before training starts,
including `Number of scaling parameters`, `N_total`, `N_scaling`, and
`flops_per_token_est`. A CPU smoke run can verify the instrumentation:

```bash
uv run python -m scripts.base_train \
  --device-type cpu \
  --depth 1 \
  --aspect-ratio 64 \
  --head-dim 64 \
  --model-dim 64 \
  --max-seq-len 128 \
  --num-iterations 0 \
  --eval-every -1 \
  --core-metric-every -1 \
  --sample-every -1 \
  --save-every -1 \
  --device-batch-size 1 \
  --total-batch-size 128
```

For full training/evaluation, download or prepare the nanochat tokenizer and
dataset in the default `NANOCHAT_BASE_DIR` cache, then run the same entrypoints
with GPU-appropriate batch sizes and horizons.
