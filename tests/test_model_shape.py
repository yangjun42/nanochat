from __future__ import annotations

import pytest

from scaling_law_harness.model_shape import resolve_model_shape


def test_resolve_model_shape_preserves_aspect_ratio_rounding_by_default() -> None:
    shape = resolve_model_shape(depth=7, aspect_ratio=72, head_dim=128)

    assert shape.model_dim == 512
    assert shape.n_head == 4
    assert shape.sizing_source == "aspect_ratio"


def test_resolve_model_shape_accepts_explicit_model_dim_override() -> None:
    shape = resolve_model_shape(depth=7, aspect_ratio=72, head_dim=128, model_dim=768)

    assert shape.model_dim == 768
    assert shape.n_head == 6
    assert shape.sizing_source == "model_dim"


def test_resolve_model_shape_rejects_model_dim_not_divisible_by_head_dim() -> None:
    with pytest.raises(ValueError, match="model_dim must be divisible by head_dim"):
        resolve_model_shape(depth=7, aspect_ratio=72, head_dim=128, model_dim=672)
