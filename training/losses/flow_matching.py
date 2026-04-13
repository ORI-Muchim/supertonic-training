"""Flow matching loss and training-time helpers for TTL (Stage 2).

Paper:
    z_t   = (1 - (1 - σ_min) · t) · z_0 + t · z_1
    target = z_1 - (1 - σ_min) · z_0
    L_CFM = E [ || m ⊙ ( v_θ(z_t, c, t) - target ) ||_1 ]

With σ_min = 0 (matches tts.json ttl.flow_matching.sig_min):
    z_t   = (1 - t) · z_0 + t · z_1
    target = z_1 - z_0

CFG: `UncondMasker` stochastically replaces conditioning with learned uncond tokens.
Batch expansion (K_e): reuse one forward pass of text/style encoders across K_e
different (z_0, t) pairs — faster WER convergence per paper.

SPFM: binary per-sample filter after warmup — train unconditionally on samples
where L_cond > L_uncond (rejects noisy labels).
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


SIGMA_MIN = 0.0     # tts.json: ttl.flow_matching.sig_min = 0


def sample_flow_matching_inputs(
    z_1: torch.Tensor,
    latent_mask: torch.Tensor | None = None,
    sigma_min: float = SIGMA_MIN,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample z_0 ~ N(0, I), t ~ U[0, 1], build z_t and velocity target.

    Args:
        z_1          : [B, C, L] ground-truth latent
        latent_mask  : [B, 1, L] optional (zeros out padded positions)
    Returns:
        (z_t, target, z_0, t) where t shape is [B]
    """
    B = z_1.shape[0]
    device = z_1.device
    z_0 = torch.randn_like(z_1)
    t = torch.rand(B, device=device)
    t_ = t.view(B, 1, 1)
    coef0 = 1.0 - (1.0 - sigma_min) * t_
    z_t = coef0 * z_0 + t_ * z_1
    target = z_1 - (1.0 - sigma_min) * z_0
    if latent_mask is not None:
        z_t = z_t * latent_mask
        target = target * latent_mask
        z_0 = z_0 * latent_mask
    return z_t, target, z_0, t


def cfm_loss(
    v_pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """L1 flow matching loss (paper uses L1).

    Args:
        v_pred : [B, C, L]
        target : [B, C, L] = z_1 - (1-σ)·z_0
        mask   : [B, 1, L] optional
    """
    err = (v_pred - target).abs()
    if mask is not None:
        err = err * mask
        if reduction == "mean":
            return err.sum() / mask.sum().clamp_min(1.0) / v_pred.shape[1]
    if reduction == "mean":
        return err.mean()
    elif reduction == "none":
        return err
    raise ValueError(reduction)


# ----------- Uncond masker (CFG dropout) -----------
@dataclass
class UncondMaskerConfig:
    prob_both_uncond: float = 0.04
    prob_text_uncond: float = 0.01
    std: float = 0.1
    text_dim: int = 256
    n_style: int = 50
    style_value_dim: int = 256


class UncondMasker(nn.Module):
    """Stochastically replace text_emb and/or style_ttl with learned uncond tokens.

    Sampling scheme per batch element:
      ε ~ U[0, 1]
      if ε < prob_both_uncond          : replace BOTH
      elif ε < prob_both_uncond + prob_text_uncond : replace TEXT only
      else                              : keep both

    Replacement = learnable_uncond + N(0, std²)
    """
    def __init__(self, cfg: UncondMaskerConfig | None = None):
        super().__init__()
        cfg = cfg or UncondMaskerConfig()
        self.cfg = cfg
        # Learnable uncond tokens
        self.uncond_text  = nn.Parameter(torch.zeros(1, cfg.text_dim, 1))                # broadcast over T
        self.uncond_style = nn.Parameter(torch.zeros(1, cfg.n_style, cfg.style_value_dim))

    def forward(self, text_emb: torch.Tensor, style_ttl: torch.Tensor):
        """Replace per-sample with uncond tokens with given probabilities.

        Returns masked (text_emb, style_ttl) with same shapes.
        """
        B = text_emb.shape[0]
        device = text_emb.device
        eps = torch.rand(B, device=device)

        p_both = self.cfg.prob_both_uncond
        p_text = self.cfg.prob_text_uncond
        drop_both = eps < p_both
        drop_text = (eps < p_both + p_text) & (~drop_both)

        if drop_both.any() or drop_text.any():
            # Build replacement tensors matching shapes
            text_unc  = self.uncond_text.expand_as(text_emb)
            style_unc = self.uncond_style.expand_as(style_ttl)
            if self.cfg.std > 0.0 and self.training:
                text_unc  = text_unc  + torch.randn_like(text_unc)  * self.cfg.std
                style_unc = style_unc + torch.randn_like(style_unc) * self.cfg.std

            # Apply per-sample replacements
            drop_text_mask  = (drop_both | drop_text).view(B, 1, 1)
            drop_style_mask = drop_both.view(B, 1, 1)
            text_emb  = torch.where(drop_text_mask,  text_unc,  text_emb)
            style_ttl = torch.where(drop_style_mask, style_unc, style_ttl)

        return text_emb, style_ttl


# ----------- Batch expander -----------
def batch_expand(*tensors: torch.Tensor, K: int) -> list[torch.Tensor]:
    """Duplicate each batch element K times along dim=0.

    Example: [B, C, T] -> [B*K, C, T] where elements are ordered
    [sample0_k0, sample0_k1, ..., sample0_k(K-1), sample1_k0, ...].
    """
    out = []
    for x in tensors:
        out.append(x.unsqueeze(1).expand(-1, K, *([-1] * (x.dim() - 1))).reshape(-1, *x.shape[1:]))
    return out


# ----------- SPFM filter -----------
def spfm_select(
    loss_cond_per_sample: torch.Tensor,
    loss_uncond_per_sample: torch.Tensor,
    step: int,
    warmup_steps: int,
) -> torch.Tensor:
    """Returns boolean mask [B]: True for samples to train CONDITIONALLY (keep c),
    False for samples that should be trained UNCONDITIONALLY (c gets replaced).

    Before warmup, always True (conditional).
    After warmup: True iff L_cond ≤ L_uncond.
    """
    if step < warmup_steps:
        return torch.ones_like(loss_cond_per_sample, dtype=torch.bool)
    return loss_cond_per_sample <= loss_uncond_per_sample


if __name__ == "__main__":
    torch.manual_seed(0)
    # Shapes: B=4, C=144 (TTL latent), L=30 (frames)
    z_1 = torch.randn(4, 144, 30)
    mask = torch.ones(4, 1, 30); mask[0, 0, 25:] = 0
    z_t, target, z_0, t = sample_flow_matching_inputs(z_1, mask)
    print(f"z_t {tuple(z_t.shape)}, target {tuple(target.shape)}, t {tuple(t.shape)}")

    v_pred = torch.randn_like(z_1)
    loss = cfm_loss(v_pred, target, mask)
    print(f"CFM loss: {loss.item():.4f}")

    # uncond masker
    text_emb  = torch.randn(4, 256, 15)
    style_ttl = torch.randn(4, 50, 256)
    um = UncondMasker()
    um.train()
    torch.manual_seed(123)
    te, st = um(text_emb, style_ttl)
    print(f"after masker: text {tuple(te.shape)}, style {tuple(st.shape)}")
    # repeated sampling to confirm probabilities
    hits_both = hits_text = hits_none = 0
    N = 2000
    for _ in range(N):
        eps = torch.rand(1).item()
        if eps < 0.04: hits_both += 1
        elif eps < 0.05: hits_text += 1
        else: hits_none += 1
    print(f"expected drop ratios in {N} samples: both={hits_both/N:.3f}, text_only={hits_text/N:.3f}, none={hits_none/N:.3f}")

    # batch expand
    a = torch.arange(3 * 2).reshape(3, 2)
    b = torch.arange(3 * 4).reshape(3, 2, 2)
    ae, be = batch_expand(a, b, K=4)
    print(f"batch_expand K=4: {tuple(a.shape)} -> {tuple(ae.shape)}, {tuple(b.shape)} -> {tuple(be.shape)}")
    assert ae.shape == (12, 2)
    # element 0..3 should all be original row 0
    assert (ae[:4] == a[0:1]).all()

    # SPFM
    lc = torch.tensor([0.1, 0.5, 0.3, 0.9])
    lu = torch.tensor([0.2, 0.3, 0.4, 0.8])
    sel = spfm_select(lc, lu, step=1000, warmup_steps=500)
    print(f"SPFM (post-warmup): {sel.tolist()}  (expected [T, F, T, F])")
