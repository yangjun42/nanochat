# Synthetic contract fixture

`create_fixture.py OUTPUT` creates a two-task campaign made entirely from fake
metadata, tiny JSON files, and plain-text “checkpoints.” It contains no real
dataset, cluster identity, run plan, budget, result, or seal decision.

Use it to exercise prepare/finalize/reduce behavior without training. The
generated files are disposable test artifacts, not scientific evidence.
The generator also creates a tiny clean Git checkout so compute-side HEAD and
dirty-tree verification can be exercised without referring to this repository.
