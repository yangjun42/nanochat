# Run Plan Schema

Required columns:

- `stage`: unique row identifier.
- `split`: `smoke`, `reference`, `heldout`, or `verification`.
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
