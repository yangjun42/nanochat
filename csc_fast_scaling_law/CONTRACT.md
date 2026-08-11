# Campaign execution contract

This is the public, experiment-agnostic boundary between a run plan and its
evidence. Private plans, real dataset locations, cluster identities, results,
and unlock decisions do not belong in this repository.

## Why this path is separate

The older train/eval array wrappers predate a strict completion contract. A
checkpoint directory can exist after an interrupted write, and a permissive
metrics merge can silently omit failed rows. `campaign_contract.py` instead
uses these rules:

1. A campaign JSON binds the exact plan, external `PLAN_FROZEN` anchor, data
   manifest, independent validation manifest, two-file tokenizer manifest,
   tokenizer-consistency receipt, environment lock, schedule, decision rules,
   and clean code commit to SHA-256 digests.
2. Every plan row records separate initialization and data-order seeds, the
   schedule ID, GPU count, and whether a target is still sealed.
3. A task writes only below `tasks/.partial/<task>/<attempt>/`. The checkpoint,
   metrics, and logs remain partial—and therefore invisible to reducers—until
   all required files pass validation.
4. Successful finalization inventories and hashes every task file, writes an
   artifact manifest and `SUCCESS`, then atomically renames the whole attempt
   to `tasks/<task>`.
5. A task is skipped only when its `SUCCESS`, manifest, plan row, provenance,
   required files, complete inventory, and hashes all validate. Any partial or
   changed directory is an error.
6. The reducer requires exactly the planned task set. A missing task, extra
   task, non-success status, missing metric, changed hash, or unauthorized
   sealed target aborts reduction before output is written.

The directory rename also makes checkpoint publication atomic. Training code
may use ordinary checkpoint writes inside the partial directory; no consumer
sees them as complete until finalization.

## Campaign schema

The top-level fields are:

```json
{
  "schema_version": "nanochat-campaign-v1",
  "campaign_id": "private-campaign-id",
  "code": {"commit": "40-or-more-lowercase-hex", "dirty": false},
  "files": {
    "plan": {"path": "...", "sha256": "..."},
    "plan_frozen": {"path": "...", "sha256": "..."},
    "data_manifest": {"path": "...", "sha256": "..."},
    "validation_manifest": {"path": "...", "sha256": "..."},
    "tokenizer_manifest": {"path": "...", "sha256": "..."},
    "tokenizer_consistency_receipt": {"path": "...", "sha256": "..."},
    "environment_lock": {"path": "...", "sha256": "..."},
    "schedule_manifest": {"path": "...", "sha256": "..."},
    "decision_rules": {"path": "...", "sha256": "..."},
    "data_preflight_receipt": {"path": "...", "sha256": "..."},
    "text_overlap_audit": {"path": "...", "sha256": "..."},
    "data_capacity_receipt": {"path": "...", "sha256": "..."}
  },
  "artifacts": {
    "root": "optional/private/output/root",
    "required": [
      {"name": "checkpoint", "glob": "work/base_checkpoints/{task_id}/model_*.pt", "min_count": 1, "max_count": 1},
      {"name": "metrics", "glob": "metrics/final.json", "min_count": 1, "max_count": 1}
    ],
    "forbid_globs": ["work/**/optim_*.pt"]
  },
  "reducer": {
    "metric_json": "metrics/final.json",
    "required_fields": ["val_bpb", "N_scaling", "D_actual"]
  }
}
```

All file paths may be absolute or relative to the campaign JSON. The public
code does not supply real values. New plans require `task_id`, `init_seed`,
`data_seed`, `schedule_id`, `gpu_count`, and `target_state`. Valid target states
are `source` and `sealed`. A provisional or unsealed target is deliberately not
an executable plan; after the decision is unlocked, create a new hash-bound
sealed plan.

A sealed row additionally requires an unlock JSON whose schema is
`nanochat-unlock-v1`. It must bind the campaign-manifest hash, plan hash,
decision-rule hash, campaign ID, `status: unlocked`, and the exact authorized
task IDs. Editing either the plan or frozen decision rules invalidates it.

Before freezing the campaign JSON, run one CPU-side full data check:

```bash
python3 -m csc_fast_scaling_law.data_preflight \
  --dataset-manifest /private/control/train.json \
  --validation-manifest /private/control/validation.json \
  --code-commit "$CLEAN_CODE_COMMIT" \
  --out /private/control/data_preflight_receipt.json
```

The receipt proves that every declared shard hash, parquet footer, non-empty
row set, required text column, and train/validation separation passed. Bind the
receipt itself in `files.data_preflight_receipt`. Each GPU task checks that
receipt, its code commit, the two manifest hashes, and current shard size/mtime;
it does not redundantly stream the full corpus once per distributed rank for
both train and eval.

Before capacity proof or GPU submission, audit the tokenizer pair. A formal
tokenizer manifest contains exactly `tokenizer.pkl` and `token_bytes.pt`:

```bash
python3 -m csc_fast_scaling_law.tokenizer_consistency \
  --tokenizer-manifest /private/control/tokenizer.json \
  --code-commit "$CLEAN_CODE_COMMIT" \
  --receipt /private/control/tokenizer_consistency_receipt.json
```

This recomputes each non-special token length from tiktoken's raw token bytes;
special-token lengths must be zero. If an old byte table is wrong, pass
`--corrected-token-bytes-out` with a new, nonexistent path, create a new sealed
tokenizer manifest around that output, and rerun the audit. Nothing is
overwritten.

## Generic Slurm flow

First validate a private manifest:

```bash
python3 -m csc_fast_scaling_law.campaign_contract validate \
  --campaign-manifest /private/control/campaign.json
```

Preview a parameterized submission (submission happens only with `--submit`):

```bash
python3 -m csc_fast_scaling_law.submit_contract_array \
  --campaign-manifest /private/control/campaign.json \
  --checkout /cluster/checkouts/nanochat-clean \
  --runner /private/control/run_one_task.sh \
  --artifact-root /cluster/evidence/campaign-id \
  --account "$CSC_ACCOUNT" \
  --project "$CSC_PROJECT" \
  --partition "$GPU_PARTITION" \
  --gpu-count 4 \
  --time 05:00:00 \
  --max-concurrent-gpus 40
```

The array throttle is computed as `floor(max_concurrent_gpus / gpu_count)`;
for four-GPU tasks this is `%10`. An optional `--concurrency` may lower it but
cannot exceed the total GPU ceiling. Sealed rows are validated against the
unlock manifest before `sbatch` is called.

`run_one_task.sh` receives the path to `prepared.json` as its only argument and
also receives `CONTRACT_*`, `NANOCHAT_BASE_DIR`, `CSC_METRICS_DIR`,
`CSC_LOG_DIR`, and `GPU_COUNT` in the environment. It must put all required
files below `CONTRACT_STAGING_DIR`; it must not create a `SUCCESS` marker.

For nanochat train-and-BPB tasks, use
`csc_fast_scaling_law/nanochat_task_runner.py` (directly in an activated
environment, or through a tiny site-specific module/venv wrapper). It accepts
only allow-listed plan columns, passes separate `--init-seed`/`--data-seed`,
validates one hash-bound preflight instead of rehashing on every rank, passes the exact
`--target-tokens`, keeps the positive reference data/parameter ratio explicit
in the plan (it must never use `-1`, because that ratio also scales weight
decay), independently evaluates every heldout file, and always uses
`--no-save-optimizer`. It writes `metrics/final.json` only after checkpoint
metadata proves that the exact requested number of prediction positions was
completed. The runner also requires a tokenizer manifest with schema version
`1` with exactly hash-bound `tokenizer.pkl` and `token_bytes.pt` entries plus a
PASS consistency receipt; it verifies both files before staging them in the
partial task directory. It also requires a single dataset key across the plan,
exact data-manifest hashes per row, a full no-reuse capacity proof for train and
every heldout shard/rank, exact per-task schedule records, FA3 at an explicitly
pinned revision, and bfloat16. Capture the environment lock only after setting
`NANOCHAT_FA3_REVISION`, prefetching that revision, and making the cache
immutable/offline for the campaign. The lock records the `kernels` version,
resolved interface origin, and hashes of loaded FA3 implementation files.
Each capacity receipt binds the SHA-256 of that campaign's single-dataset
submission CSV—not a mixed global planning table. `PLAN_FROZEN` separately
anchors the projection back to the frozen global decision artifact.

At task preparation and again before promotion, the compute node runs Git
checks against the actual checkout. `HEAD` must equal `code.commit`, porcelain
status (including untracked files) must be empty, and recursive submodules must
be initialized at their recorded commits. The observed HEAD, tree, clean-status
hash, and submodule-status hash are stored in each task artifact manifest.

After all tasks finish, reduction is intentionally strict:

```bash
python3 -m csc_fast_scaling_law.campaign_contract reduce \
  --campaign-manifest /private/control/campaign.json \
  --artifact-root /cluster/evidence/campaign-id \
  --out /private/control/reduced_metrics.json
```

See `examples/synthetic_contract/create_fixture.py` for a fake, zero-training
fixture. Its “checkpoint” is plain synthetic text and must never be interpreted
as a model or research result.
