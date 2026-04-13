"""Vector field wrapper with classifier-free guidance (CFG) baked into forward.

The original vector_estimator.onnx runs ONE Euler step of the conditional velocity
field. For high quality, the paper trains with an uncond_masker (CFG dropout, 4%/1%
probability) and at inference combines:
    v_cfg = (1 + w) · v_cond − w · v_uncond
with w = cfg_scale (paper default: 3).

This wrapper:
  • Stores uncond tokens (learned in uncond_masker during Stage 2) as buffers.
  • At forward, runs the backbone velocity TWICE (cond + uncond) and combines.
  • Emits the same-shape output as the non-CFG version.
  • Adds one extra input: `cfg_scale [B]` (set 0 to disable CFG, matching old behavior).

Runtime cost: ~2× the non-CFG vector_estimator forward.  Still fast on 3090.

ONNX export handles buffers as initializers, so the uncond tokens get baked into
the exported graph — no need for separate weight files at inference.
"""
from __future__ import annotations
import os, sys
from pathlib import Path

import torch
import torch.nn as nn

# Reuse the verified VectorField from analysis/
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "analysis"))
from torch_vector_estimator import VectorField    # type: ignore


class VectorFieldCFG(nn.Module):
    """CFG-aware Euler step.

    Inputs:
        noisy_latent  [B, 144, L]
        text_emb      [B, 256, T]
        style_ttl     [B, 50, 256]
        latent_mask   [B, 1, L]
        text_mask     [B, 1, T]
        current_step  [B]
        total_step    [B]
        cfg_scale     [B]   — per-sample guidance weight (0 disables CFG)
    Output:
        denoised_latent [B, 144, L]
    """
    def __init__(
        self,
        vector_field: VectorField,
        uncond_text:  torch.Tensor,    # [1, 256, 1]  (broadcast over T at runtime)
        uncond_style: torch.Tensor,    # [1, 50, 256]
    ):
        super().__init__()
        self.vf = vector_field
        assert uncond_text.shape == (1, 256, 1),  f"uncond_text shape {uncond_text.shape}"
        assert uncond_style.shape == (1, 50, 256), f"uncond_style shape {uncond_style.shape}"
        # Non-persistent buffer? No — we WANT these in the ONNX as initializers.
        self.register_buffer("uncond_text",  uncond_text.float())
        self.register_buffer("uncond_style", uncond_style.float())

    def forward(
        self,
        noisy_latent: torch.Tensor,
        text_emb:     torch.Tensor,
        style_ttl:    torch.Tensor,
        latent_mask:  torch.Tensor,
        text_mask:    torch.Tensor,
        current_step: torch.Tensor,
        total_step:   torch.Tensor,
        cfg_scale:    torch.Tensor,
    ) -> torch.Tensor:
        B = noisy_latent.shape[0]
        T_text = text_emb.shape[-1]

        # Normalized time
        t_norm = current_step / total_step

        # Conditional velocity
        v_cond = self.vf.velocity(
            noisy_latent, text_emb, style_ttl, latent_mask, text_mask, t_norm
        )

        # Build uncond conditioning by broadcasting learned tokens
        text_u  = self.uncond_text.expand(B, -1, T_text)      # [B, 256, T_text]
        style_u = self.uncond_style.expand(B, -1, -1)         # [B, 50, 256]

        # Unconditional velocity
        v_uncond = self.vf.velocity(
            noisy_latent, text_u, style_u, latent_mask, text_mask, t_norm
        )

        # CFG combination (vectorized per-sample)
        w = cfg_scale.view(-1, 1, 1)
        v_cfg = (1.0 + w) * v_cond - w * v_uncond

        # Euler ODE step
        dt = (1.0 / total_step).view(-1, 1, 1)
        denoised = (noisy_latent + dt * v_cfg) * latent_mask
        return denoised


def load_cfg_from_ttl_checkpoint(
    ttl_ckpt_path: str | Path,
    device: torch.device,
) -> VectorFieldCFG:
    """Build a VectorFieldCFG from a TTL training checkpoint.

    Expects checkpoint dict keys: 'vector_field', 'uncond_masker'.
    uncond_masker state has: 'uncond_text' [1, 256, 1], 'uncond_style' [1, 50, 256].
    """
    ck = torch.load(str(ttl_ckpt_path), map_location=device, weights_only=False)
    if "vector_field" not in ck:
        raise KeyError(f"'vector_field' not in checkpoint keys: {list(ck.keys())}")
    if "uncond_masker" not in ck:
        raise KeyError(f"'uncond_masker' not in checkpoint keys: {list(ck.keys())}")

    vf = VectorField().to(device)
    vf.load_state_dict(ck["vector_field"])
    vf.eval()

    um_state = ck["uncond_masker"]
    uncond_text  = um_state["uncond_text"]    # [1, 256, 1]
    uncond_style = um_state["uncond_style"]   # [1, 50, 256]

    model = VectorFieldCFG(vf, uncond_text, uncond_style).to(device)
    model.eval()
    return model


if __name__ == "__main__":
    # Smoke test: random weights; verify shapes match non-CFG version.
    torch.manual_seed(0)
    vf = VectorField()
    uncond_text  = torch.zeros(1, 256, 1)
    uncond_style = torch.zeros(1, 50, 256)
    m = VectorFieldCFG(vf, uncond_text, uncond_style)
    m.eval()
    B, L, T = 2, 17, 25
    inputs = dict(
        noisy_latent=torch.randn(B, 144, L),
        text_emb=torch.randn(B, 256, T),
        style_ttl=torch.randn(B, 50, 256),
        latent_mask=torch.ones(B, 1, L),
        text_mask=torch.ones(B, 1, T),
        current_step=torch.tensor([1.0, 2.0]),
        total_step=torch.tensor([5.0, 5.0]),
        cfg_scale=torch.tensor([3.0, 3.0]),
    )
    with torch.no_grad():
        out = m(**inputs)
    print(f"VectorFieldCFG output: {tuple(out.shape)}  stats: min={out.min():.3f} max={out.max():.3f}")
    # Sanity: cfg_scale=0 should match non-CFG vector_field forward
    inputs["cfg_scale"] = torch.zeros(B)
    with torch.no_grad():
        out_no_cfg = m(**inputs)
        out_ref    = vf(
            inputs["noisy_latent"], inputs["text_emb"], inputs["style_ttl"],
            inputs["latent_mask"], inputs["text_mask"],
            inputs["current_step"], inputs["total_step"],
        )
    d = (out_no_cfg - out_ref).abs().max().item()
    print(f"cfg_scale=0 vs non-CFG forward max|Δ|: {d:.4e}  (should be 0)")
    assert d < 1e-5
    print("OK: CFG wrapper preserves non-CFG behavior when cfg_scale=0.")
