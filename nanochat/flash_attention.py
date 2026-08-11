"""
Unified Flash Attention interface with automatic FA3/SDPA switching.

Exports `flash_attn` module that matches the FA3 API exactly, but falls back
to PyTorch SDPA on non-Hopper GPUs (including Blackwell), MPS, and CPU.

Usage (drop-in replacement for FA3):
    from nanochat.flash_attention import flash_attn

    # Training (no KV cache)
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

    # Inference (with KV cache)
    y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
"""
import os
import torch
import torch.nn.functional as F


# =============================================================================
# Detection: Try to load FA3 on Hopper+ GPUs
# =============================================================================
def _load_flash_attention_3():
    """Try to load Flash Attention 3 (requires Hopper GPU, sm90)."""
    if not torch.cuda.is_available():
        return None
    local_dir = os.environ.get("NANOCHAT_FA3_LOCAL_DIR")
    try:
        major, _ = torch.cuda.get_device_capability()
        # FA3 kernels are compiled for Hopper (sm90) only
        # Ada (sm89), Blackwell (sm100) need SDPA fallback until FA3 is recompiled
        if major != 9:
            return None
        import importlib
        from pathlib import Path
        import sys

        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        revision = os.environ.get("NANOCHAT_FA3_REVISION")

        # A content-addressed local variant is the preferred formal path. It
        # avoids the authenticated HF kernels API and remains usable offline.
        # The environment lock later binds the resolved revision and hashes
        # every loaded .py/.so file.
        if local_dir:
            root = Path(local_dir).expanduser().resolve(strict=True)
            if not revision or revision not in root.parts:
                raise RuntimeError(
                    "NANOCHAT_FA3_LOCAL_DIR must be nested below the pinned revision"
                )
            path_entry = str(root)
            sys.path.insert(0, path_entry)
            try:
                packaged_interface = (
                    root / "flash_attention_3" / "flash_attn_interface.py"
                )
                source_build_interface = root / "flash_attn_interface.py"
                if packaged_interface.is_file():
                    module_name = "flash_attention_3.flash_attn_interface"
                elif source_build_interface.is_file() and (
                    root / "flash_attn_3"
                ).is_dir():
                    module_name = "flash_attn_interface"
                else:
                    raise RuntimeError(
                        "NANOCHAT_FA3_LOCAL_DIR has no supported FA3 interface layout"
                    )
                interface = importlib.import_module(module_name)
                if module_name == "flash_attn_interface":
                    # The native wheel imports its generated top-level config
                    # lazily from round_up_headdim(), after this loader returns.
                    # Load it while the isolated artifact root is still visible.
                    supporting_modules = [
                        importlib.import_module("flash_attn_config"),
                        importlib.import_module("flash_attn_3._C"),
                    ]
                else:
                    supporting_modules = [
                        importlib.import_module("flash_attention_3.flash_attn_config"),
                        importlib.import_module("flash_attention_3._C"),
                    ]
                for module in [interface, *supporting_modules]:
                    module_origin = Path(module.__file__).resolve(strict=True)
                    module_origin.relative_to(root)
            finally:
                if path_entry in sys.path:
                    sys.path.remove(path_entry)
            origin = Path(interface.__file__).resolve(strict=True)
            origin.relative_to(root)
            artifact_files = []
            for path in sorted(root.rglob("*")):
                if path.is_symlink():
                    raise RuntimeError(f"FA3 local package contains a symlink: {path}")
                if path.is_file() and path.suffix in {".py", ".so"}:
                    artifact_files.append(str(path.resolve(strict=True)))
            if not artifact_files or str(origin) not in artifact_files:
                raise RuntimeError("FA3 local package has no hashable interface artifact")
            interface.__fa3_artifact_files__ = tuple(artifact_files)
            return interface

        from kernels import get_kernel
        kwargs = {"revision": revision} if revision else {}
        # kernels>=0.14 requires an explicit trust decision for community
        # repositories. Formal runs additionally require a full pinned Git
        # revision and bind the resolved revision plus every loaded file hash in
        # the environment lock before training is allowed.
        return get_kernel(
            'varunneal/flash-attention-3',
            trust_remote_code=True,
            **kwargs,
        ).flash_attn_interface
    except Exception:
        # An explicit formal local artifact is a contract, not a preference.
        # Missing, mismatched, or unloadable files must not silently select SDPA.
        if local_dir:
            raise
        return None


_fa3 = _load_flash_attention_3()
HAS_FA3 = _fa3 is not None

# Override for testing: set to 'fa3', 'sdpa', or None (auto)
_override_impl = None


def _resolve_use_fa3():
    """Decide once whether to use FA3, based on availability, override, and dtype."""
    if _override_impl == 'fa3':
        assert HAS_FA3, "Cannot override to FA3: not available on this hardware"
        return True
    if _override_impl == 'sdpa':
        return False
    if HAS_FA3:
        # FA3 Hopper kernels only support bf16 and fp8; fp16/fp32 must use SDPA fallback
        from nanochat.common import COMPUTE_DTYPE
        if COMPUTE_DTYPE == torch.bfloat16:
            return True
        return False
    return False

USE_FA3 = _resolve_use_fa3()


# =============================================================================
# SDPA helpers
# =============================================================================
def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    SDPA attention with sliding window support.
    q, k, v are (B, H, T, D) format.
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    # Full context, same length
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    # Single token generation
    if Tq == 1:
        if window >= 0 and window < Tk:
            # window is "left" tokens we need to include (window + 1) keys total
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # Need explicit mask for sliding window/chunk inference
    device = q.device
    # For chunk inference (Tq != Tk), is_causal is not aligned to cache position => build an explicit bool mask
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx

    # sliding window (left)
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)

# =============================================================================
# Public API: Same interface as FA3
# =============================================================================
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Flash Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    if USE_FA3:
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    # SDPA fallback: transpose (B, T, H, D) -> (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)  # back to (B, T, H, D)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """
    Flash Attention with KV cache for inference.

    FA3 updates k_cache/v_cache in-place. Our SDPA fallback does the same.

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    if USE_FA3:
        return _fa3.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )

    # SDPA fallback: manually manage KV cache
    B, T_new, H, D = q.shape
    pos = cache_seqlens[0].item()  # assume uniform position across batch

    # Insert new k, v into cache (in-place, matching FA3 behavior)
    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    # Get full cache up to current position + new tokens
    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    # Transpose to SDPA layout: (B, T, H, D) -> (B, H, T, D)
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)

    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)

    return y_sdpa.transpose(1, 2)  # back to (B, T, H, D)


# =============================================================================
# Export: flash_attn module interface (drop-in replacement for FA3)
# =============================================================================
from types import SimpleNamespace
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
