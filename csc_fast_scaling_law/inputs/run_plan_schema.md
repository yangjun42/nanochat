# Run Plan Schema

Required columns:

- `stage`: unique row identifier.
- `split`: `smoke`, `reference`, `heldout`, or `final_verify`.
- `recipe_band`: usually `canonical`; use another value only when intentionally changing the recipe.
- `depth`: nanochat model depth.
- `seed`: training seed.
- `num_iterations`: base training iterations.
- `target_tokens`: intended training tokens, used as `D_actual` if logs do not override it.
- `total_batch_size`, `device_batch_size`, `max_seq_len`, `warmup_steps`, `fp8`, `window_pattern`: nanochat training arguments.
- `model_tag`: checkpoint tag.

Optional columns are preserved in merged metrics when present, but they are not
required by the artifact contract. The reference-grid plan uses:

- `n_index`, `d_index`: intended grid coordinates before measured `N_scaling`
  is known.
- `planned_family`: descriptive plan family, for example
  `roihu_reference_depth_token_grid`.
- `planned_model_depth`, `planned_data_tokens`, `planned_total_batch_size`:
  planned controls used to create the row. The measured `N_scaling` and
  `D_actual` in merged metrics remain the authoritative fitting coordinates.
- `aspect_ratio`, `head_dim`: optional architecture controls passed to
  `scripts.base_train`. When omitted, the runner uses nanochat defaults
  `aspect_ratio=64` and `head_dim=128`. Use
  `python -m csc_fast_scaling_law.model_size_probe` to choose candidate
  architectures before submitting GPU jobs.
- `model_dim`: optional explicit embedding width passed to `scripts.base_train`.
  When present, it must be divisible by `head_dim` and takes precedence over
  the default `ceil(depth * aspect_ratio / head_dim) * head_dim` sizing rule.

Tracked example plans:

- `smoke_plan.csv`: two-row train/eval harness smoke check.
- `roihu_reference_3x3_seed0.csv`: first 3x3 reference grid after smoke.
- `roihu_reference_depthspread_3x3_seed0.csv`: second 3x3 reference grid
  using depths `1,4,8` after the first real slice showed depth `1,2,4`
  gives uneven measured `N_scaling` spacing.
- `roihu_reference_sizeprobe_3x3_seed0.csv`: preflight-selected 3x3
  candidate using depths `2,3,7`, `aspect_ratio=48`, and `head_dim=128`
  from the non-training size probe.
- `roihu_reference_sizeprobe72_3x3_seed0.csv`: next preflight-selected 3x3
  candidate using depths `2,4,7`, `aspect_ratio=72`, and `head_dim=128`.
  The fixed-control model-size scan estimates a much smaller geometric spacing
  error before training, but its `is_exact_geometric` probe flag is still
  `False`.
- `roihu_reference_exactprobe_3x3_seed0.csv`: exact-geometric preflight 3x3
  candidate using explicit model-axis controls
  `(depth=4, aspect_ratio=1, head_dim=64)`,
  `(depth=2, aspect_ratio=49, head_dim=32)`, and
  `(depth=1, aspect_ratio=225, head_dim=32)`. Its preflight `N_scaling`
  values have exact ratios `2.0` and `2.0`; smoke this unusual control grid
  before a full run.
- `roihu_reference_sizeprobe72_finalverify_seed1_2.csv`: final-verification
  replicate of the recommended fixed-control path. It reruns the sizeprobe72
  `d_index=2` model competitors at seeds `1` and `2`, and includes source
  campaign/stage/loss metadata plus `verification_role`.
- `roihu_reference_sizeprobe72_depth47_finalverify_seed3_5.csv`:
  seed-extension final-verification plan for the unresolved same-data
  depth-4/depth-7 contenders. It keeps `d_index=2`, `aspect_ratio=72`,
  `head_dim=128`, and 256 training iterations, and runs seeds `3`, `4`, and
  `5`.
- `roihu_reference_sizeprobe72_depth47_finalverify_seed6_10.csv`:
  next seed-extension final-verification block for the same depth-4/depth-7
  contenders. It keeps the same `d_index=2`, `aspect_ratio=72`, `head_dim=128`,
  and 256-iteration controls, and adds paired seeds `6` through `10`.
- `roihu_reference_sizeprobe72_source_replicate_seed11_12.csv`:
  full source-surface replication plan for the sizeprobe72 3x3 grid. It keeps
  `split=reference`, repeats all nine `n_index,d_index` cells at seeds `11` and
  `12`, and uses `replicate_source_*` metadata instead of
  `source_campaign_id`/`verification_role` so these rows are not consumed as
  final-verification referee rows.
- `roihu_reference_controlled_modeldim_hd64_smoke_seed0.csv`: three-row smoke
  slice for the adviser-requested controlled-geometry line. It fixes
  `depth=4`, `head_dim=64`, `window_pattern=L`, and `d_index=0`, while varying
  explicit `model_dim=448,640,896`.
- `roihu_reference_controlled_modeldim_hd64_3x3_seed0.csv`: full 3x3
  controlled-geometry follow-up plan using the same fixed controls and explicit
  `model_dim` values. Submit this only after the smoke slice verifies measured
  `N_scaling` and artifact merging.
- `roihu_reference_controlled_modeldim_hd64_5p_fd_smoke_seed1_3.csv`: paired
  5-point finite-difference smoke for the controlled-geometry strict-gate
  blocker. It keeps only `(n0,d0)`, `(n1,d0)`, `(n2,d0)`, `(n0,d1)`, and
  `(n0,d2)` for seeds `1` through `3`, so the next Roihu spend checks
  positive/shrinking BPB deltas before a full rerun.
- `roihu_reference_controlled_modeldim_hd64_low_overhead_5p_seed21.csv`:
  single-seed 5-point source stencil for overhead accounting. It uses the same
  five controlled-model-dim cells as the finite-difference smoke, but only seed
  `21`, so the next Roihu spend can test whether a minimal estimator source
  plan is cheaper than one best measured target-line run.
