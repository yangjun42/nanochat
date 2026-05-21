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

## Smoke Flow

```bash
cd /scratch/project_2017828/$USER/nanochat

sbatch --array=1-2%1 csc_fast_scaling_law/run_train_array.sbatch

# After training succeeds, reuse the same CAMPAIGN_ID/NANOCHAT_BASE_DIR:
sbatch --array=1-2%1 csc_fast_scaling_law/run_eval_array.sbatch
```

Use `csc_fast_scaling_law/monitor_roihu.sh` from a local machine to inspect
queue state, recent accounting, and the latest merged metrics.
