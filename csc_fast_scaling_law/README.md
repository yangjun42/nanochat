# Fast Scaling-Law Roihu Harness

This folder contains the Roihu execution harness for the fast scaling-law
validation project. It intentionally lives in the nanochat fork because it runs
real nanochat training and evaluation jobs. The analysis notebooks and paper
figures live in the separate `fast-scaling-law-fitting-experiments` repository.

## Contract

The harness consumes a run-plan CSV with one row per `(N, D)` configuration and
writes:

- `metrics.csv`: merged plan, train, eval, size, loss, and cost metadata.
- `metrics.jsonl`: the same merged rows as JSON lines.
- `fit_ready_cells.csv`: rows with `N_scaling`, `D_actual`, and `val_bpb_final`.

These files are the artifact boundary consumed by the scaling-law experiment
repo. Raw logs and checkpoints should stay on Roihu or in ignored output
folders.

The scripts prepend `csc_fast_scaling_law/wandb_stub` to `PYTHONPATH` because the
Roihu PyTorch container can contain an unrelated broken `wandb`/protobuf
combination. This keeps `--run=dummy` jobs independent of system `wandb`.

## Model-Size Preflight

Before submitting a new scaling-law grid, probe candidate nanochat architectures
locally or on Roihu without training:

```bash
python -m csc_fast_scaling_law.model_size_probe \
  --depths 1-16 \
  --aspect-ratios 48,64,80,96,112,128 \
  --max-n-scaling 50000000 \
  --out /tmp/nanochat_fsl_size_probe.csv \
  --triplet-out /tmp/nanochat_fsl_triplets.csv \
  --print-triplet
```

The printed triplet is a non-degenerate, approximately geometric `N_scaling`
proposal under the chosen size bound. Add its `depth`, `aspect_ratio`, and
`head_dim` columns to the run-plan CSV; `run_train_array.sbatch` passes these
controls through to `scripts.base_train`. The measured training log still
remains authoritative for `N_scaling`. The triplet CSV includes
`is_exact_geometric` and `geometric_cross_product_residual`; strict closed-form
`5pEst` candidates require the former to be `True`, not merely a small spacing
error.

A wider fixed-control scan found the next candidate plan
`inputs/roihu_reference_sizeprobe72_3x3_seed0.csv`:

- depths `2,4,7`;
- `aspect_ratio=72`;
- `head_dim=128`;
- preflight `N_scaling` values `9,961,496`, `19,660,872`, `38,797,504`;
- preflight geometric relative error about `0.000178`;
- `is_exact_geometric=False` with integer cross-product residual
  `68,706,894,400`.

An expanded formula-only exact search over depths `1-32`, aspect ratios
`1-256`, and head dimensions `32,64,96,128,160,192,224,256` found the tracked
candidate `inputs/roihu_reference_exactprobe_3x3_seed0.csv`:

- model-axis controls
  `(depth=4, aspect_ratio=1, head_dim=64)`,
  `(depth=2, aspect_ratio=49, head_dim=32)`,
  `(depth=1, aspect_ratio=225, head_dim=32)`;
- preflight `N_scaling` values `2,293,784`, `4,587,568`, `9,175,136`;
- exact preflight multipliers `2.0` and `2.0`;
- `is_exact_geometric=True` with integer cross-product residual `0`.

Because this candidate changes multiple architecture controls across the model
axis, run a smoke slice before a full 3x3 submission and treat measured
training logs as authoritative.

The current revised real-validation path is the tracked final-verification plan
`inputs/roihu_reference_sizeprobe72_finalverify_seed1_2.csv`. It keeps the
fixed-control sizeprobe72 architecture (`aspect_ratio=72`, `head_dim=128`) and
reruns the largest-data (`d_index=2`) competitors at depths `2,4,7` for seeds
`1` and `2`. The seed-0 middle-model winner is marked with
`verification_role=observed_best`; the endpoints are
`same_data_competitor`. Judge this plan by held-out BPB/final verification, not
by treating a fitted surface as its own referee.

## Smoke Flow

```bash
cd /scratch/project_2017828/$USER/nanochat

sbatch --array=1-2%1 csc_fast_scaling_law/run_train_array.sbatch

# After training succeeds, reuse the same CAMPAIGN_ID/NANOCHAT_BASE_DIR:
sbatch --array=1-2%1 csc_fast_scaling_law/run_eval_array.sbatch
```

Use `csc_fast_scaling_law/monitor_roihu.sh` from a local machine to inspect
queue state, recent accounting, and the latest merged metrics.

## Sequential Fallback

If Slurm rejects an array with `AssocMaxSubmitJobLimit`, use the sequential
wrappers for small grids. They submit a single Slurm job and run the selected
plan rows one after another on the same allocation while reusing the same
per-row array wrappers:

```bash
CAMPAIGN_ID=fsl-sizeprobe72-3x3-YYYYMMDD-HHMM \
PLAN_CSV=/scratch/project_2017828/$USER/nanochat/csc_fast_scaling_law/inputs/roihu_reference_sizeprobe72_3x3_seed0.csv \
NANOCHAT_BASE_DIR=/scratch/project_2017828/$USER/nanochat-cache/$CAMPAIGN_ID \
sbatch --partition=gpuinteractive --time=02:00:00 \
  csc_fast_scaling_law/run_train_sequence.sbatch

# After training succeeds:
CAMPAIGN_ID=fsl-sizeprobe72-3x3-YYYYMMDD-HHMM \
PLAN_CSV=/scratch/project_2017828/$USER/nanochat/csc_fast_scaling_law/inputs/roihu_reference_sizeprobe72_3x3_seed0.csv \
NANOCHAT_BASE_DIR=/scratch/project_2017828/$USER/nanochat-cache/$CAMPAIGN_ID \
sbatch --partition=gpuinteractive --time=01:00:00 \
  csc_fast_scaling_law/run_eval_sequence.sbatch
```

Set `ROW_START` and `ROW_END` to run a slice, for example `ROW_START=1
ROW_END=3`. Keep temporary plan CSVs on shared scratch or in the checkout, not
login-node `/tmp`, because compute nodes cannot see login-local files.
