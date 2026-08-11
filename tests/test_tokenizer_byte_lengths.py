from __future__ import annotations

import tiktoken
import torch

from nanochat.tokenizer import RustBPETokenizer


def _raw_byte_tokenizer() -> RustBPETokenizer:
    encoding = tiktoken.Encoding(
        name="synthetic-raw-bytes",
        pat_str=r"(?s:.)",
        mergeable_ranks={bytes([value]): value for value in range(256)},
        special_tokens={"<|bos|>": 256},
    )
    return RustBPETokenizer(encoding, "<|bos|>")


def test_token_byte_lengths_use_raw_bytes_not_utf8_replacement_text() -> None:
    tokenizer = _raw_byte_tokenizer()

    lengths = tokenizer.token_byte_lengths()

    assert lengths.dtype == torch.int32
    assert tuple(lengths.shape) == (257,)
    assert lengths[0].item() == 1
    assert lengths[128].item() == 1
    assert len(tokenizer.decode([128]).encode("utf-8")) == 3
    assert lengths[255].item() == 1
    assert lengths[tokenizer.get_bos_token_id()].item() == 0
