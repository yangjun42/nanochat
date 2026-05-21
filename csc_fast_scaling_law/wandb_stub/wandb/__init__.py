"""Tiny wandb stub for Roihu jobs that run nanochat with --run=dummy.

The CSC PyTorch container can have a broken system wandb/protobuf combination.
nanochat imports wandb before it decides to use DummyWandb, so this import-only
stub prevents unrelated wandb import failures without changing nanochat code.
"""


class _Run:
    def __init__(self, *args, **kwargs):
        self.config = kwargs.get("config", {})

    def log(self, *args, **kwargs):
        return None

    def finish(self, *args, **kwargs):
        return None


def init(*args, **kwargs):
    return _Run(*args, **kwargs)


def log(*args, **kwargs):
    return None


def finish(*args, **kwargs):
    return None
