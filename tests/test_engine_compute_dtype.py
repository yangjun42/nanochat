from __future__ import annotations

import pytest
import torch

from nanochat import engine, flash_attention, gpt


class _GenerationTokenizer:
    # Keep control tokens outside the tiny model's output vocabulary so the
    # test necessarily exercises both prompt prefill and multiple decode steps.
    special_tokens = (
        "<|python_start|>", "<|python_end|>", "<|output_start|>",
        "<|output_end|>", "<|assistant_end|>", "<|bos|>",
    )

    def encode_special(self, token):
        return 128 + self.special_tokens.index(token)

    def get_bos_token_id(self):
        return self.encode_special("<|bos|>")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_engine_prefill_and_decode_support_compute_dtype_overrides(monkeypatch, dtype):
    monkeypatch.setattr(engine, "COMPUTE_DTYPE", dtype)
    monkeypatch.setattr(gpt, "COMPUTE_DTYPE", dtype)
    monkeypatch.setattr(flash_attention, "USE_FA3", False)
    torch.manual_seed(42)
    model = gpt.GPT(gpt.GPTConfig(
        sequence_len=16, vocab_size=128, n_layer=1,
        n_head=1, n_kv_head=1, n_embd=32, window_pattern="L",
    ))
    model.init_weights()
    model.eval()

    columns = list(engine.Engine(model, _GenerationTokenizer()).generate(
        [1, 2], num_samples=2, max_tokens=3, temperature=0.0,
    ))

    assert len(columns) == 3
    assert all(len(tokens) == 2 and masks == [1, 1] for tokens, masks in columns)
    assert all(0 <= token < 128 for tokens, _ in columns for token in tokens)
