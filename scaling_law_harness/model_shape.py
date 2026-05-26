from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelShape:
    depth: int
    aspect_ratio: int
    head_dim: int
    model_dim: int
    n_head: int
    sizing_source: str


def resolve_model_shape(
    *,
    depth: int,
    aspect_ratio: int,
    head_dim: int,
    model_dim: int | None = None,
) -> ModelShape:
    """Resolve nanochat width/head counts from the default or explicit sizing rule."""

    if min(int(depth), int(aspect_ratio), int(head_dim)) <= 0:
        raise ValueError("depth, aspect_ratio, and head_dim must be positive")

    depth = int(depth)
    aspect_ratio = int(aspect_ratio)
    head_dim = int(head_dim)
    if model_dim is not None and int(model_dim) > 0:
        resolved_model_dim = int(model_dim)
        if resolved_model_dim % head_dim != 0:
            raise ValueError("model_dim must be divisible by head_dim")
        sizing_source = "model_dim"
    else:
        base_dim = depth * aspect_ratio
        resolved_model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
        sizing_source = "aspect_ratio"

    n_head = resolved_model_dim // head_dim
    if n_head <= 0:
        raise ValueError("resolved model_dim must produce at least one attention head")
    return ModelShape(
        depth=depth,
        aspect_ratio=aspect_ratio,
        head_dim=head_dim,
        model_dim=resolved_model_dim,
        n_head=n_head,
        sizing_source=sizing_source,
    )
