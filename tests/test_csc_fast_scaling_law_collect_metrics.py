from __future__ import annotations

import csv
import json
from pathlib import Path

from csc_fast_scaling_law.collect_metrics import collect_campaign


def test_collect_campaign_writes_fit_ready_cells(tmp_path: Path) -> None:
    plan = tmp_path / "plan.csv"
    with plan.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stage",
                "recipe_band",
                "depth",
                "seed",
                "num_iterations",
                "target_tokens",
                "total_batch_size",
                "device_batch_size",
                "max_seq_len",
                "warmup_steps",
                "fp8",
                "window_pattern",
                "model_tag",
                "split",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "stage": "smoke_d2_i64_s0",
                "recipe_band": "canonical",
                "depth": "2",
                "seed": "0",
                "num_iterations": "64",
                "target_tokens": "33554432",
                "total_batch_size": "524288",
                "device_batch_size": "16",
                "max_seq_len": "2048",
                "warmup_steps": "40",
                "fp8": "1",
                "window_pattern": "L",
                "model_tag": "fsl-smoke",
                "split": "smoke",
            }
        )

    train_metrics = tmp_path / "train_metrics"
    eval_metrics = tmp_path / "eval_metrics"
    train_logs = tmp_path / "train_logs"
    eval_logs = tmp_path / "eval_logs"
    train_metrics.mkdir()
    eval_metrics.mkdir()
    train_logs.mkdir()
    eval_logs.mkdir()

    (train_logs / "smoke_d2_i64_s0.log").write_text(
        "N_total: 123456\nN_scaling: 45678\nflops_per_token_est: 987654\ntrain bpb: 1.23\n",
        encoding="utf-8",
    )
    (eval_logs / "bpb_smoke_d2_i64_s0.log").write_text("val bpb: 1.11\n", encoding="utf-8")
    (train_metrics / "stage_metrics.jsonl").write_text(
        json.dumps(
            {
                "stage": "smoke_d2_i64_s0",
                "elapsed_seconds": 12,
                "exit_code": 0,
                "partition": "gpumedium",
                "log_path": str(train_logs / "smoke_d2_i64_s0.log"),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (eval_metrics / "stage_metrics.jsonl").write_text(
        json.dumps(
            {
                "stage": "bpb_smoke_d2_i64_s0",
                "elapsed_seconds": 5,
                "exit_code": 0,
                "partition": "gpumedium",
                "log_path": str(eval_logs / "bpb_smoke_d2_i64_s0.log"),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    outputs = collect_campaign(
        plan_csv=plan,
        train_metrics_dir=train_metrics,
        eval_metrics_dir=eval_metrics,
        out_dir=tmp_path / "out",
    )

    assert outputs["metrics_csv"].exists()
    assert outputs["metrics_jsonl"].exists()
    assert outputs["fit_ready_cells_csv"].exists()

    rows = list(csv.DictReader(outputs["fit_ready_cells_csv"].open(newline="", encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["stage"] == "smoke_d2_i64_s0"
    assert rows[0]["N_scaling"] == "45678.0"
    assert rows[0]["D_actual"] == "33554432.0"
    assert rows[0]["val_bpb_final"] == "1.11"
    assert rows[0]["C_6ND"] == str(6.0 * 45678.0 * 33554432.0)
    assert rows[0]["gpu_seconds_train"] == "48.0"


def test_collect_campaign_parses_current_base_train_parameter_table(tmp_path: Path) -> None:
    plan = tmp_path / "plan.csv"
    with plan.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stage",
                "split",
                "recipe_band",
                "depth",
                "seed",
                "num_iterations",
                "target_tokens",
                "total_batch_size",
                "device_batch_size",
                "max_seq_len",
                "warmup_steps",
                "fp8",
                "window_pattern",
                "model_tag",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "stage": "current_log_shape",
                "split": "smoke",
                "recipe_band": "canonical",
                "depth": "4",
                "seed": "0",
                "num_iterations": "64",
                "target_tokens": "33554432",
                "total_batch_size": "524288",
                "device_batch_size": "16",
                "max_seq_len": "2048",
                "warmup_steps": "40",
                "fp8": "1",
                "window_pattern": "L",
                "model_tag": "fsl-current-log",
            }
        )

    train_metrics = tmp_path / "train_metrics"
    eval_metrics = tmp_path / "eval_metrics"
    train_logs = tmp_path / "train_logs"
    eval_logs = tmp_path / "eval_logs"
    train_metrics.mkdir()
    eval_metrics.mkdir()
    train_logs.mkdir()
    eval_logs.mkdir()

    (train_logs / "current_log_shape.log").write_text(
        "\n".join(
            [
                "Parameter counts:",
                "total                   : 1,000,000",
                "transformer_matrices    : 700,000",
                "lm_head                 : 300,000",
                "Estimated FLOPs per token: 1.234000e+06",
                "Minimum validation bpb: 1.234500",
            ]
        ),
        encoding="utf-8",
    )
    (eval_logs / "bpb_current_log_shape.log").write_text("val bpb: 1.111000\n", encoding="utf-8")
    (train_metrics / "stage_metrics.jsonl").write_text(
        json.dumps({"stage": "current_log_shape", "elapsed_seconds": 10, "exit_code": 0})
        + "\n",
        encoding="utf-8",
    )
    (eval_metrics / "stage_metrics.jsonl").write_text(
        json.dumps({"stage": "bpb_current_log_shape", "elapsed_seconds": 4, "exit_code": 0})
        + "\n",
        encoding="utf-8",
    )

    outputs = collect_campaign(
        plan_csv=plan,
        train_metrics_dir=train_metrics,
        eval_metrics_dir=eval_metrics,
        out_dir=tmp_path / "out",
    )

    rows = list(csv.DictReader(outputs["fit_ready_cells_csv"].open(newline="", encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["N_total"] == "1000000.0"
    assert rows[0]["N_scaling"] == "1000000.0"
    assert rows[0]["flops_per_token_est"] == "1234000.0"
    assert rows[0]["train_bpb_final"] == "1.2345"
    assert rows[0]["val_bpb_final"] == "1.111"


def test_slurm_wrappers_use_local_wandb_stub() -> None:
    root = Path(__file__).resolve().parents[1]
    train_script = (root / "csc_fast_scaling_law" / "run_train_array.sbatch").read_text(encoding="utf-8")
    eval_script = (root / "csc_fast_scaling_law" / "run_eval_array.sbatch").read_text(encoding="utf-8")
    stub = root / "csc_fast_scaling_law" / "wandb_stub" / "wandb" / "__init__.py"

    assert "csc_fast_scaling_law/wandb_stub" in train_script
    assert "csc_fast_scaling_law/wandb_stub" in eval_script
    assert stub.exists()
    assert "def init" in stub.read_text(encoding="utf-8")
