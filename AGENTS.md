# nanochat fork: instructions for AI agents

Training harness for scaling-law experiments, forked from upstream nanochat.
`origin` is a public fork: everything pushed there is public.

## Layout and commands

- Working branch: `notification/202608-runtime-contracts`. `master` follows
  upstream.
- `csc_fast_scaling_law/` holds run plans, input manifests and the launchers
  for the cluster. These are execution contracts, not results.
- Dependencies and run rules are in the README section "Runtime and
  artifacts". The standard check is
  `uv run --locked --extra cpu --group dev pytest -q -m "not slow"`.
- Datasets, checkpoints and logs are runtime state outside this folder.

## Working rules

- Start with `git log -5` and `git status`. Another agent may have worked here.
- Frozen campaign snapshots are copied unchanged when a result must be
  comparable with earlier runs. A change to the trainer needs a repeated run
  that reproduces an earlier result before new results are compared with old.
- Push branches only. Push a tag only when its name and message are neutral;
  other local tags stay local.
- Keep machine paths, account names and unpublished results out of commits.
- Finish with a commit that says what changed.
