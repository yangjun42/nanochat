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

Completed campaign `fsl-sizeprobe72-finalverify-seq-20260522-0037` ran all
six train rows and all six BPB eval rows successfully on `gpumedium`. Seed 1
confirms the depth-4 candidate (`val_bpb=1.142690`), but seed 2 favors depth 7
(`val_bpb=1.097519`). The two-seed mean final BPB is depth 2: `1.230689`,
depth 4: `1.164960`, and depth 7: `1.146944`. Treat the sizeprobe72 path as
still referee-ready, but do not lock the middle-depth allocation without more
seeded final verification against the depth-7 same-data competitor.

The next tracked plan is
`inputs/roihu_reference_sizeprobe72_depth47_finalverify_seed3_5.csv`. It keeps
the same largest-data fixed-control setting and reruns only the unsettled
depth-4/depth-7 contenders at seeds `3`, `4`, and `5`.

Completed campaign `fsl-sizeprobe72-depth47-finalverify-seq-20260522-0110`
ran all six train rows and all six BPB eval rows successfully on `gpumedium`.
Seeds 3 and 4 favor depth 7 (`val_bpb=1.090444` and `1.104864`), while seed 5
favors depth 4 (`val_bpb=1.142550` versus depth 7 at `1.200685`). The
seed-extension mean final BPB is depth 4: `1.143270` and depth 7: `1.131998`;
across all final-verification seeds 1-5, depth 7 is the current mean leader
(`1.137976` versus depth 4 at `1.151946`) and wins 3/5 paired seeds. Keep the
held-out-BPB referee active before treating the allocation as fully settled,
because the depth-7 seed variance remains visibly larger.

Completed campaign
`fsl-sizeprobe72-depth47-finalverify-seed6-10-seq-20260522-0640` ran the next
tracked execution block,
`inputs/roihu_reference_sizeprobe72_depth47_finalverify_seed6_10.csv`, with all
ten train rows and all ten BPB eval rows completing successfully on
`gpumedium`. Seed 6 favored depth 7, while seeds 7-10 favored depth 4. Across
paired final-verification seeds 1-10, depth 4 is confirmed by the 10-seed
referee (`1.155772` mean BPB versus depth 7 at `1.161703`) and wins 6/10 paired
seeds. This confirms the source observed-grid depth-4 decision under the
held-out BPB/final-verification referee, but it does not turn the gate-blocked
strict real Delta-Ensemble path into a proven real-data algorithm claim.

The next tracked source-surface run is
`inputs/roihu_reference_sizeprobe72_source_replicate_seed11_12.csv`. It keeps the
fixed-control sizeprobe72 3x3 grid (`aspect_ratio=72`, `head_dim=128`) and
repeats all nine source cells at seeds `11` and `12` with `split=reference`.
The rows carry `replicate_source_*` metadata rather than
`source_campaign_id`/`verification_role`, so the analysis repo treats them as a
new source surface for cell-mean strict-gate and Delta-Ensemble eligibility
checks, not as final-verification referee rows.

The controlled `model_dim` geometry line has three tracked plans:

- `inputs/roihu_reference_controlled_modeldim_hd64_smoke_seed0.csv`: a 3-row
  smoke slice at `d_index=0` with `depth=4`, `head_dim=64`, and explicit
  `model_dim=448,640,896`. Use this first to validate `--model-dim` train,
  eval, and merge artifacts.
- `inputs/roihu_reference_controlled_modeldim_hd64_3x3_seed0.csv`: the full
  3x3 follow-up grid. Run it only after the smoke slice verifies measured
  `N_scaling` and artifact merging.
- `inputs/roihu_reference_controlled_modeldim_hd64_5p_fd_smoke_seed1_3.csv`:
  a paired-seed 5-point finite-difference smoke for the strict-gate blocker.
  It runs `(n0,d0)`, `(n1,d0)`, `(n2,d0)`, `(n0,d1)`, and `(n0,d2)` for seeds
  `1` through `3` to test whether positive/shrinking BPB deltas stabilize
  before spending another full controlled-model-dim grid.
- `inputs/roihu_reference_controlled_modeldim_hd64_low_overhead_5p_seed21.csv`:
  a single-seed 5-point source stencil for overhead accounting. It uses the
  same cells as the finite-difference smoke, but only seed `21`, to measure
  whether the cheapest practical estimator source plan can fall below the best
  measured target-line run before scaling to paired seeds.
- `inputs/roihu_source_prefix_hd32_optgeo_5x5_seed26.csv` and
  `inputs/roihu_source_prefix_hd32_optgeo_5x5_seed27.csv`: paired nested
  5x5 source-prefix sweeps for the paper-facing Roihu estimator curves. They
  fix `depth=4`, `head_dim=32`, `window_pattern=L`, and explicit
  `model_dim=224,320,448,608,832`, giving `N_scaling` values from about
  `9.75M` to `60.49M` with improved adjacent geometric error while covering
  the measured target-line optimum from both sides.

## Smoke Flow

```bash
cd /scratch/project_2017828/$USER/nanochat

sbatch --array=1-2%1 csc_fast_scaling_law/run_train_array.sbatch

# After training succeeds, reuse the same CAMPAIGN_ID/NANOCHAT_BASE_DIR:
sbatch --array=1-2%1 csc_fast_scaling_law/run_eval_array.sbatch
```

Use `csc_fast_scaling_law/monitor_roihu.sh` from a local machine to inspect
queue state, recent accounting, and the latest merged metrics.
The monitor runs a local CSC SSH user-certificate preflight before any remote
SSH command. Override the certificate path with `ROIHU_SSH_CERT` when needed;
an expired certificate fails locally with the expiry timestamp instead of
falling through to a remote `Permission denied (publickey)` error.

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
