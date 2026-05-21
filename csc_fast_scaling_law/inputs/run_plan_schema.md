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
required by the artifact contract.

