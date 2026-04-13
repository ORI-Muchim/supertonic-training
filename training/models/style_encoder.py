"""Style encoders: produce compact 'reference' representations from AE latent.

Two variants from tts.json:
  TTL style encoder → [B, 50, 256]   (used by text_encoder + vector_estimator as V)
  DP  style encoder → [B,  8,  16]   (used by duration_predictor)

Both follow this sketch:
    latent [B, 24, T]                    (from frozen AE encoder)
      → chunk_compress 6× → [B, 144, T/6]
      → proj_in Conv1d 144 → hdim
      → N × ConvNeXt (symmetric edge pad, ksz=5)
      → StyleTokenLayer   (learnable-query cross-attn pool to [B, n_style, value_dim])
      → (n_refine - 1) × CrossAttentionRefine  (tokens re-attend to audio + FFN)
      → output [B, n_style, value_dim]

Paper (arXiv 2503.23108) quotes:
  "50 learnable vectors with a dimension of 128 are used in the first attention block"
    → matches n_heads=2 × head_dim=128 = 256 attn dim in the FIRST block.
  "outputs reference key and value vectors through two attention layers"
    → two attention layers. The K output is absorbed as a learnable prototype that
       lives inside vector_estimator / text_encoder (we verified this during RE:
       vector_estimator's `/Expand_output_0 [1, 50, 256]` and text_encoder's baked
       `tanh(prototype)` are precisely these K tensors). Our style encoder only
       needs to emit V (the per-speaker `style_ttl`) at inference time.

We therefore implement 2 stacked cross-attention layers by default (n_refine=2).
A single-layer variant is kept as a safety fallback via `n_refine=1`.

DP config has `style_key_dim=0` which we interpret as "no separate key projection"
(shared with value): fallback `key_dim = value_dim`.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvNeXt1D, symmetric_pad


def chunk_compress(z: torch.Tensor, kc: int) -> torch.Tensor:
    """Inverse of vocoder un-chunk.  [B, C, T] → [B, C*kc, T/kc] with sub-pixel layout:
    sample_{t*kc+r} of channel c in input → (c*kc + r, t) in output.
    """
    B, C, T = z.shape
    if T % kc != 0:
        # Pad right so T divisible by kc (replicate last frame).
        pad = kc - (T % kc)
        z = F.pad(z, (0, pad), mode="replicate")
        T = z.shape[-1]
    z = z.reshape(B, C, T // kc, kc).permute(0, 1, 3, 2).contiguous()   # [B, C, kc, T/kc]
    z = z.reshape(B, C * kc, T // kc)
    return z


def chunk_uncompress(z: torch.Tensor, C: int, kc: int) -> torch.Tensor:
    """Inverse of chunk_compress (for completeness)."""
    B, Ckc, Tc = z.shape
    assert Ckc == C * kc
    z = z.reshape(B, C, kc, Tc).permute(0, 1, 3, 2).contiguous()
    return z.reshape(B, C, Tc * kc)


class StyleTokenLayer(nn.Module):
    """Learnable-query attention pool (GST-inspired).

    Inputs : audio features [B, T, input_dim]  (or [B, input_dim, T] auto-transpose)
    Outputs: style tokens   [B, n_style, value_dim]
    """
    def __init__(
        self,
        input_dim: int,
        n_style: int,
        value_dim: int,
        n_heads: int,
        prototype_dim: int,
        key_dim: int | None = None,
    ):
        super().__init__()
        # Treat key_dim=0 (DP config quirk) as "use value_dim for keys".
        if key_dim is None or key_dim == 0:
            key_dim = value_dim
        assert key_dim % n_heads == 0 and value_dim % n_heads == 0, \
            f"key_dim={key_dim}, value_dim={value_dim} must be divisible by n_heads={n_heads}"
        self.n_style = n_style
        self.n_heads = n_heads
        self.head_dim_k = key_dim // n_heads
        self.head_dim_v = value_dim // n_heads
        self.value_dim  = value_dim

        # Learnable style prototypes (queries). Broadcast over batch.
        self.style_tokens = nn.Parameter(torch.zeros(1, n_style, prototype_dim))
        nn.init.normal_(self.style_tokens, std=0.02)

        self.W_q = nn.Linear(prototype_dim, key_dim)
        self.W_k = nn.Linear(input_dim, key_dim)
        self.W_v = nn.Linear(input_dim, value_dim)
        self.W_o = nn.Linear(value_dim, value_dim)

    def forward(self, feats: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Expected layout: [B, C, T] (NCL, matching ConvNeXt1D output). Transpose to [B, T, C] internally.
        feats = feats.transpose(1, 2)
        B, T, C = feats.shape
        H = self.n_heads

        q = self.W_q(self.style_tokens.expand(B, -1, -1))  # [B, n_style, kd]
        k = self.W_k(feats)                                 # [B, T, kd]
        v = self.W_v(feats)                                 # [B, T, vd]

        q = q.reshape(B, self.n_style, H, self.head_dim_k).transpose(1, 2)  # [B,H,n_style,Dk]
        k = k.reshape(B, T, H, self.head_dim_k).transpose(1, 2)             # [B,H,T,Dk]
        v = v.reshape(B, T, H, self.head_dim_v).transpose(1, 2)             # [B,H,T,Dv]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim_k)
        if attn_mask is not None:
            # attn_mask [B, 1, 1, T] or broadcastable
            scores = scores.masked_fill(attn_mask == 0, -1e4)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)                         # [B,H,n_style,Dv]
        out = out.transpose(1, 2).reshape(B, self.n_style, self.value_dim)
        return self.W_o(out)


class CrossAttentionRefine(nn.Module):
    """A second (+) attention block: the `style tokens` cross-attend to the same audio
    features and get refined. Includes LN + FFN (transformer-decoder style).

    Inputs:
        tokens  [B, n_style, value_dim]   — from previous attention layer
        audio   [B, C, T]                 — same audio features StyleTokenLayer used
    Output:
        refined tokens [B, n_style, value_dim]
    """
    def __init__(self, input_dim: int, value_dim: int, n_heads: int, key_dim: int | None = None,
                 ffn_ratio: int = 4):
        super().__init__()
        if key_dim is None or key_dim == 0:
            key_dim = value_dim
        assert value_dim % n_heads == 0 and key_dim % n_heads == 0
        self.value_dim = value_dim
        self.n_heads = n_heads
        self.head_dim_v = value_dim // n_heads
        self.head_dim_k = key_dim // n_heads

        self.norm_q = nn.LayerNorm(value_dim, eps=1e-6)
        self.W_q = nn.Linear(value_dim, key_dim)
        self.W_k = nn.Linear(input_dim, key_dim)
        self.W_v = nn.Linear(input_dim, value_dim)
        self.W_o = nn.Linear(value_dim, value_dim)

        self.norm_ffn = nn.LayerNorm(value_dim, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(value_dim, value_dim * ffn_ratio),
            nn.GELU(),
            nn.Linear(value_dim * ffn_ratio, value_dim),
        )

    def forward(self, tokens: torch.Tensor, audio: torch.Tensor,
                attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        # tokens: [B, N, Cv];  audio: [B, Ci, T] → transpose to [B, T, Ci]
        audio = audio.transpose(1, 2)
        B, N, Cv = tokens.shape
        T = audio.shape[1]
        H = self.n_heads

        # Cross-attention
        q_norm = self.norm_q(tokens)
        q = self.W_q(q_norm).reshape(B, N, H, self.head_dim_k).transpose(1, 2)  # [B,H,N,Dk]
        k = self.W_k(audio).reshape(B, T, H, self.head_dim_k).transpose(1, 2)   # [B,H,T,Dk]
        v = self.W_v(audio).reshape(B, T, H, self.head_dim_v).transpose(1, 2)   # [B,H,T,Dv]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim_k)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e4)
        attn = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, self.value_dim)
        tokens = tokens + self.W_o(attn_out)

        # FFN
        tokens = tokens + self.ffn(self.norm_ffn(tokens))
        return tokens


class StyleEncoder(nn.Module):
    """Reference encoder: AE latent [B, 24, T] → style tokens [B, n_style, value_dim].

    Stack of `n_refine` attention layers:
      layer 1: StyleTokenLayer (learnable queries → audio K/V)
      layer 2..n_refine: CrossAttentionRefine (tokens → audio K/V, with FFN + residual)
    """
    def __init__(
        self,
        ldim: int = 24,
        chunk_compress_factor: int = 6,
        hdim: int = 256,
        intermediate_dim: int = 1024,
        num_layers: int = 6,
        ksz: int = 5,
        n_style: int = 50,
        value_dim: int = 256,
        key_dim: int | None = 256,
        prototype_dim: int = 256,
        n_heads: int = 2,
        n_refine: int = 2,            # total attention layers (paper: "two attention layers")
        pad_mode: str = "symmetric",
        out_scale: float = 0.0625,    # shipped voice_styles all have std=0.0625 exactly.
                                       # Post-LayerNorm × this scale fixes our encoder's
                                       # output distribution to match the pre-trained
                                       # TE/VF's expected style_ttl scale.
    ):
        super().__init__()
        assert n_refine >= 1
        self.ldim = ldim
        self.kc = chunk_compress_factor
        self.proj_in = nn.Conv1d(ldim * chunk_compress_factor, hdim, 1, bias=False)
        self.convnext = nn.ModuleList([
            ConvNeXt1D(hdim, intermediate_dim, ksz, 1, pad_mode=pad_mode)
            for _ in range(num_layers)
        ])
        self.style_token_layer = StyleTokenLayer(
            input_dim=hdim, n_style=n_style, value_dim=value_dim,
            n_heads=n_heads, prototype_dim=prototype_dim, key_dim=key_dim,
        )
        self.refine_layers = nn.ModuleList([
            CrossAttentionRefine(input_dim=hdim, value_dim=value_dim,
                                 n_heads=n_heads, key_dim=key_dim)
            for _ in range(n_refine - 1)
        ])
        # Output whitening + fixed scale. Keeps encoder output distribution
        # in the regime that shipped TE/VF were trained on (otherwise loss is
        # dominated by a constant scale mismatch and the model can't converge).
        self.out_norm = nn.LayerNorm(value_dim)
        self.register_buffer("out_scale", torch.tensor(float(out_scale)))

    def forward(self, latent: torch.Tensor, frame_mask: torch.Tensor | None = None) -> torch.Tensor:
        """latent [B, 24, T] → [B, n_style, value_dim]."""
        x = chunk_compress(latent, self.kc)   # [B, 144, T/kc]
        x = self.proj_in(x)                    # [B, hdim, T/kc]
        for blk in self.convnext:
            x = blk(x)

        # Attention mask compressed from AE frame mask
        attn_mask = None
        if frame_mask is not None:
            Bm, _, Tm = frame_mask.shape
            if Tm % self.kc != 0:
                frame_mask = F.pad(frame_mask, (0, self.kc - (Tm % self.kc)), value=0)
            compressed = frame_mask.reshape(Bm, 1, -1, self.kc).max(dim=-1).values  # [B,1,T/kc]
            attn_mask = compressed.unsqueeze(2)   # [B, 1, 1, T/kc]

        tokens = self.style_token_layer(x, attn_mask=attn_mask)     # [B, N, V]
        for refine in self.refine_layers:
            tokens = refine(tokens, x, attn_mask=attn_mask)
        # Normalize per-token dims then scale to the shipped distribution.
        return self.out_norm(tokens) * self.out_scale


class StyleEncoderTTL(StyleEncoder):
    """50 tokens × 256 dim, 2 attention layers (paper-faithful)."""
    def __init__(self, n_refine: int = 2):
        super().__init__(
            ldim=24, chunk_compress_factor=6,
            hdim=256, intermediate_dim=1024, num_layers=6, ksz=5,
            n_style=50, value_dim=256, key_dim=256,
            prototype_dim=256, n_heads=2, n_refine=n_refine,
        )


class StyleEncoderDP(StyleEncoder):
    """8 tokens × 16 dim. DP config doesn't specify n_refine — we default to 2 as well."""
    def __init__(self, n_refine: int = 2):
        super().__init__(
            ldim=24, chunk_compress_factor=6,
            hdim=64, intermediate_dim=256, num_layers=4, ksz=5,
            n_style=8, value_dim=16, key_dim=0,    # config has key_dim=0 → handled as value_dim
            prototype_dim=64, n_heads=2, n_refine=n_refine,
        )


if __name__ == "__main__":
    for name, Cls in [("TTL", StyleEncoderTTL), ("DP", StyleEncoderDP)]:
        enc = Cls()
        n = sum(p.numel() for p in enc.parameters())
        # 2 sec @ 44.1kHz → 2*44100/512 ≈ 173 AE frames. Must be divisible by 6 → pad to 174.
        latent = torch.randn(3, 24, 173)
        out = enc(latent)
        print(f"{name}: params={n/1e6:.2f}M  latent {tuple(latent.shape)} -> style {tuple(out.shape)}")
