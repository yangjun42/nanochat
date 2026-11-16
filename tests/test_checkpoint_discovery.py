from __future__ import annotations

import pytest

from nanochat.checkpoint_manager import find_last_step


@pytest.mark.parametrize(
    "filenames, expected",
    [
        (["model_000009.pt", "model_000010.pt"], 10),
        (["model_9.pt", "model_10.pt"], 10),
        (["model_999999.pt", "model_1000000.pt"], 1_000_000),
    ],
)
def test_checkpoint_steps_are_ordered_numerically(tmp_path, filenames, expected):
    for filename in filenames:
        (tmp_path / filename).touch()
    assert find_last_step(tmp_path) == expected


def test_checkpoint_discovery_ignores_other_files_and_directories(tmp_path):
    for filename in (
        "model_000010.pt",
        "model_backup.pt",
        "model_000020_rank0.pt",
        "backup_model_000030.pt",
        "model_000040.pt.tmp",
    ):
        (tmp_path / filename).touch()
    (tmp_path / "model_000050.pt").mkdir()
    assert find_last_step(tmp_path) == 10


def test_checkpoint_discovery_reports_no_valid_checkpoint(tmp_path):
    (tmp_path / "model_backup.pt").touch()
    with pytest.raises(FileNotFoundError, match="No checkpoints found"):
        find_last_step(tmp_path)
