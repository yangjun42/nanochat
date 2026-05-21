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
  --print-triplet
```

The printed triplet is a non-degenerate, approximately geometric `N_scaling`
proposal under the chosen size bound. Add its `depth`, `aspect_ratio`, and
`head_dim` columns to the run-plan CSV; `run_train_array.sbatch` passes these
controls through to `scripts.base_train`. The measured training log still
remains authoritative for `N_scaling`.

## Smoke Flow

```bash
cd /scratch/project_2017828/$USER/nanochat

sbatch --array=1-2%1 csc_fast_scaling_law/run_train_array.sbatch

# After training succeeds, reuse the same CAMPAIGN_ID/NANOCHAT_BASE_DIR:
sbatch --array=1-2%1 csc_fast_scaling_law/run_eval_array.sbatch
```

Use `csc_fast_scaling_law/monitor_roihu.sh` from a local machine to inspect
queue state, recent accounting, and the latest merged metrics.
