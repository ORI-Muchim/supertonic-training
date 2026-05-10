"""PyTorch reimplementation of vector_estimator.onnx — the flow-matching vector field.

Structure (24 main_blocks = 4 outer-blocks × 6 submodules, confirmed from ONNX init names):
  Inputs:
    noisy_latent [B, 144, L], text_emb [B, 256, T], style_ttl [B, 50, 256],
    latent_mask [B, 1, L], text_mask [B, 1, T],
    current_step [B], total_step [B]

  Flow:
    t = current_step / total_step                                 # [B]
    time_emb = mlp(sinusoidal(t, 64))                              # [B, 64]  (mlp: Linear+Mish+Linear)
    x = proj_in(noisy_latent) * latent_mask                        # [B, 512, L] (Conv1d 144→512 ksz=1)
    for i in 0..3:                                                  # 4 outer blocks
        x = convnext_4L(x, mask)    # dilations [1,2,4,8]
        x = (x + time_film_i(time_emb)[:, :, None]) * mask         # time FiLM (shift only)
        x = convnext_1L(x, mask)
        x = x + larope_text_attn(x, text_emb, latent_mask, text_mask)  # residual IN attn
        x = convnext_1L(x, mask)
        x = x + style_attn(x, style_ttl, latent_mask)              # residual IN attn
    x = last_convnext_4L(x, mask)                                   # dilations [1,1,1,1]
    x = proj_out(x) * latent_mask                                   # Conv1d 512→144 ksz=1
    return x
"""
from __future__ import annotations
import math, os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================== basic blocks ===========================
def sym_pad_rep(x, ksz, dilation=1):
    pad = (ksz - 1) * dilation // 2
    return F.pad(x, (pad, pad), mode="replicate")


class ConvNeXtLayerMasked(nn.Module):
    def __init__(self, dim, inter, ksz, dilation):
        super().__init__()
        self.ksz, self.dilation = ksz, dilation
        self.dwconv = nn.Conv1d(dim, dim, ksz, groups=dim, dilation=dilation, bias=True)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Conv1d(dim, inter, 1, bias=True)
        self.pwconv2 = nn.Conv1d(inter, dim, 1, bias=True)
        self.gamma = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x, mask):
        x = x * mask
        dw = self.dwconv(sym_pad_rep(x, self.ksz, self.dilation))
        dw = dw * mask
        h = self.norm(dw.transpose(1, 2)).transpose(1, 2)
        h = self.pwconv1(h)
        h = F.gelu(h, approximate="none")
        h = self.pwconv2(h)
        h = self.gamma * h
        out = (x + h) * mask
        return out


class ConvNeXtStack(nn.Module):
    def __init__(self, dim, inter, ksz, dilations):
        super().__init__()
        self.layers = nn.ModuleList([ConvNeXtLayerMasked(dim, inter, ksz, d) for d in dilations])

    def forward(self, x, mask):
        for lyr in self.layers:
            x = lyr(x, mask)
        return x


# =========================== time encoder ===========================
class SinusoidalEmbedding(nn.Module):
    """64-dim sinusoidal time embedding."""
    def __init__(self, dim=64):
        super().__init__()
        half = dim // 2
        # Supertonic uses freqs[j] = 10000^(-j/(half-1)), range 1.0 → 1e-4 (verified via ONNX Constant_3).
        freqs = torch.tensor([10000 ** (-j / (half - 1)) for j in range(half)], dtype=torch.float32)
        self.register_buffer("freqs", freqs.view(1, half))

    def forward(self, t):
        """t: [B]. Returns [B, dim]. Supertonic scales t by 1000 before sinusoidal (matches ONNX Const_2=1000)."""
        angle = (t.unsqueeze(-1) * 1000.0) * self.freqs
        emb = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)
        return emb


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class TimeEncoder(nn.Module):
    """Mish MLP: 64 -> 256 -> 64."""
    def __init__(self, time_dim=64, hdim=256):
        super().__init__()
        self.sinusoidal = SinusoidalEmbedding(time_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, hdim),
            Mish(),
            nn.Linear(hdim, time_dim),
        )

    def forward(self, t):
        """t: [B]. Returns [B, time_dim]."""
        emb = self.sinusoidal(t)
        return self.mlp(emb)


# ============================ FiLM ================================
class TimeFiLM(nn.Module):
    def __init__(self, time_dim=64, feat_dim=512):
        super().__init__()
        self.linear = nn.Linear(time_dim, feat_dim, bias=True)

    def forward(self, x, time_emb, mask):
        """x: [B, feat_dim, L], time_emb: [B, time_dim], mask: [B, 1, L]."""
        shift = self.linear(time_emb).unsqueeze(-1)  # [B, feat_dim, 1]
        return (x + shift) * mask


# ============================ LARoPE ================================
class LARoPETextCrossAttention(nn.Module):
    """Cross-attn with Length-Aware RoPE on Q and K.
    Q from latent [B, 512, L] -> Linear 512->256, 4 heads × 64 head_dim.
    K, V from text_emb [B, 256, T] -> Linear 256->256.
    theta = 10 * 10000^(-2j/64) (rotary_scale=10 baked into theta).
    out_fc: Linear 256->512; output added to input latent (residual stored in ONNX via main_blocks.3 Add).
    """
    def __init__(self, dim=512, attn_dim=256, ctx_dim=256, n_heads=4, max_pos=1000):
        super().__init__()
        assert attn_dim % n_heads == 0
        self.dim, self.attn_dim, self.ctx_dim = dim, attn_dim, ctx_dim
        self.n_heads = n_heads
        self.head_dim = attn_dim // n_heads
        self.half = self.head_dim // 2
        self.W_query = nn.Linear(dim, attn_dim, bias=True)
        self.W_key   = nn.Linear(ctx_dim, attn_dim, bias=True)
        self.W_value = nn.Linear(ctx_dim, attn_dim, bias=True)
        self.out_fc  = nn.Linear(attn_dim, dim, bias=True)
        # Length-aware RoPE frequency table. The released ONNX stores this as a
        # constant: theta_j = 10 * 10000^(-2j/head_dim). Keep the same default
        # for from-scratch training; load_ve_weights overwrites it for shipped
        # compatibility.
        theta = 10.0 * (10000.0 ** (-torch.arange(self.half, dtype=torch.float32) / float(self.half)))
        self.register_buffer("theta", theta.view(1, 1, self.half))
        # positions 0..max_pos-1
        self.register_buffer("increments", torch.arange(max_pos, dtype=torch.float32).view(1, max_pos, 1))
        self.max_pos = max_pos

    @staticmethod
    def _lengths_from_mask(mask):
        """mask [B, 1, L] -> length[B]"""
        return mask.sum(dim=(-2, -1))

    def _rope_angles(self, seq_len, lengths):
        """lengths [B], seq_len int. Returns cos/sin [B, seq_len, half]."""
        pos = self.increments[:, :seq_len, :]                 # [1, seq_len, 1]
        # angle = (pos / length[B]) * theta[half]
        # shape: [B, seq_len, 1] * [1, 1, half] -> [B, seq_len, half]
        inv_len = (1.0 / lengths).view(-1, 1, 1)
        angle = (pos * inv_len) * self.theta                  # [B, seq_len, half]
        return torch.cos(angle), torch.sin(angle)

    @staticmethod
    def _apply_rope(x, cos, sin):
        """x: [B, H, S, D], cos/sin: [B, S, D/2] -> rotate halves concat.
        out = concat(x1*cos - x2*sin, x1*sin + x2*cos) where (x1, x2) = x.split(half)."""
        x1, x2 = x.chunk(2, dim=-1)       # each [B, H, S, half]
        cos = cos.unsqueeze(1)            # [B, 1, S, half]
        sin = sin.unsqueeze(1)
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return torch.cat([out1, out2], dim=-1)

    def forward(self, x, ctx, lat_mask, ctx_mask):
        """x: [B, 512, L], ctx: [B, 256, T], lat_mask [B,1,L], ctx_mask [B,1,T]."""
        B, _, L = x.shape
        T = ctx.shape[-1]
        x_T = x.transpose(1, 2)     # [B, L, 512]
        ctx_T = ctx.transpose(1, 2) # [B, T, 256]

        q = self.W_query(x_T)       # [B, L, 256]
        k = self.W_key(ctx_T)       # [B, T, 256]
        v = self.W_value(ctx_T)     # [B, T, 256]
        q = q.reshape(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # LARoPE
        lat_lengths = self._lengths_from_mask(lat_mask)
        ctx_lengths = self._lengths_from_mask(ctx_mask)
        q_cos, q_sin = self._rope_angles(L, lat_lengths)
        k_cos, k_sin = self._rope_angles(T, ctx_lengths)
        q = self._apply_rope(q, q_cos, q_sin)
        k = self._apply_rope(k, k_cos, k_sin)

        # scores = Q @ K^T / sqrt(dim=attn_dim=256) = /16
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attn_dim)
        # mask (via Where-like: set softmax to 0 for padded k positions)
        mask_k = ctx_mask.unsqueeze(1)            # [B, 1, 1, T]
        scores = scores.masked_fill(mask_k == 0, -1e4)
        attn = F.softmax(scores, dim=-1)
        # also zero out padded queries after softmax (doesn't affect result much, matches ONNX)
        mask_q = lat_mask.unsqueeze(1).transpose(-1, -2)  # [B, 1, L, 1]
        attn = attn * mask_q
        ctx_out = torch.matmul(attn, v)           # [B, H, L, D]
        merged = ctx_out.permute(0, 2, 1, 3).reshape(B, L, self.attn_dim)
        out = self.out_fc(merged)                 # [B, L, 512]
        out = out.transpose(1, 2) * lat_mask      # [B, 512, L]
        return out


class StyleCrossAttention(nn.Module):
    """Style cross-attn: K = tanh(W_key(prototype)) from a fixed learnable [1, N, dim],
    V = W_value(style_ttl). 2 heads × 128 head_dim. Scale = sqrt(attn_dim) = 16.

    The `prototype` is shared across all outer blocks (stored once in ONNX as /Expand_output_0).
    """
    def __init__(self, dim=512, attn_dim=256, style_dim=256, n_heads=2, n_style=50):
        super().__init__()
        assert attn_dim % n_heads == 0
        self.dim, self.attn_dim = dim, attn_dim
        self.n_heads = n_heads
        self.head_dim = attn_dim // n_heads
        self.n_style = n_style
        self.W_query = nn.Linear(dim, attn_dim, bias=True)
        self.W_key   = nn.Linear(style_dim, attn_dim, bias=True)
        self.W_value = nn.Linear(style_dim, attn_dim, bias=True)
        self.out_fc  = nn.Linear(attn_dim, dim, bias=True)

    def forward(self, x, style, lat_mask, prototype):
        """x: [B, dim, L], style: [B, N, style_dim], lat_mask [B, 1, L], prototype [1, N, style_dim]."""
        B, _, L = x.shape
        N = self.n_style
        x_T = x.transpose(1, 2)               # [B, L, dim]
        # K source is the fixed prototype (broadcast batch), NOT style_ttl
        proto = prototype.expand(B, -1, -1)   # [B, N, style_dim]
        q = self.W_query(x_T)                  # [B, L, attn_dim]
        k = self.W_key(proto)                  # [B, N, attn_dim]
        v = self.W_value(style)                # [B, N, attn_dim]
        q = q.reshape(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # tanh on K (matches ONNX runtime Tanh on Transpose_output_0)
        k = torch.tanh(k)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attn_dim)
        attn = F.softmax(scores, dim=-1)
        mask_q = lat_mask.unsqueeze(1).transpose(-1, -2)  # [B, 1, L, 1]
        attn = attn * mask_q
        ctx_out = torch.matmul(attn, v)
        merged = ctx_out.permute(0, 2, 1, 3).reshape(B, L, self.attn_dim)
        out = self.out_fc(merged)
        return out.transpose(1, 2) * lat_mask


# ======================= main block container =======================
class MainBlock(nn.Module):
    """One outer block = 6 submodules (indexed as main_blocks.6i..6i+5 in ONNX)."""
    def __init__(self, dim=512, time_dim=64, inter=1024, ksz=5,
                 conv_dils_0=(1, 2, 4, 8), conv_dils_1=(1,), conv_dils_2=(1,),
                 text_dim=256, text_heads=4,
                 style_dim=256, style_heads=2):
        super().__init__()
        self.convnext_0 = ConvNeXtStack(dim, inter, ksz, conv_dils_0)     # main_blocks.6i+0
        self.time_film  = TimeFiLM(time_dim, dim)                         # main_blocks.6i+1
        self.convnext_1 = ConvNeXtStack(dim, inter, ksz, conv_dils_1)     # main_blocks.6i+2
        self.text_attn  = LARoPETextCrossAttention(dim=dim, ctx_dim=text_dim, n_heads=text_heads)  # .6i+3
        self.convnext_2 = ConvNeXtStack(dim, inter, ksz, conv_dils_2)     # main_blocks.6i+4
        self.style_attn = StyleCrossAttention(dim=dim, style_dim=style_dim, n_heads=style_heads)    # .6i+5
        # LN layers for attn residual norms
        self.text_norm  = nn.LayerNorm(dim, eps=1e-6)   # main_blocks.6i+3.norm
        self.style_norm = nn.LayerNorm(dim, eps=1e-6)   # main_blocks.6i+5.norm

    def forward(self, x, time_emb, text_emb, style, lat_mask, text_mask, style_prototype):
        x = self.convnext_0(x, lat_mask)
        x = self.time_film(x, time_emb, lat_mask)
        x = self.convnext_1(x, lat_mask)
        # text cross-attn with residual and LN
        attn_out = self.text_attn(x, text_emb, lat_mask, text_mask)
        x = x + attn_out
        x = self.text_norm(x.transpose(1, 2)).transpose(1, 2) * lat_mask
        x = self.convnext_2(x, lat_mask)
        # style cross-attn with residual and LN (prototype is shared learnable)
        attn_out = self.style_attn(x, style, lat_mask, style_prototype)
        x = x + attn_out
        x = self.style_norm(x.transpose(1, 2)).transpose(1, 2) * lat_mask
        return x


# ========================== top-level ==========================
class VectorField(nn.Module):
    def __init__(self, dim=512, latent_dim=144, n_outer=4, time_dim=64,
                 inter=1024, ksz=5, text_dim=256, style_dim=256,
                 learn_style_prototype: bool = True):
        super().__init__()
        self.time_encoder = TimeEncoder(time_dim, 256)
        self.proj_in = nn.Conv1d(latent_dim, dim, 1, bias=False)
        self.main_blocks = nn.ModuleList([MainBlock(dim=dim, time_dim=time_dim, inter=inter, ksz=ksz,
                                                    text_dim=text_dim, style_dim=style_dim) for _ in range(n_outer)])
        self.last_convnext = ConvNeXtStack(dim, inter, ksz, (1, 1, 1, 1))
        self.proj_out = nn.Conv1d(dim, latent_dim, 1, bias=False)
        # learnable style prototype (replaces style_ttl as K source for all style_attn).
        # Zero-init kills gradient flow via tanh saturation check — use small random so
        # K = tanh(W_key(proto)) has non-trivial grad w.r.t. proto from step 0.
        # (This only affects from-scratch training; loading shipped ONNX weights
        #  overrides with the trained prototype, so bit-close verification is unaffected.)
        if learn_style_prototype:
            self.style_prototype = nn.Parameter(torch.randn(1, 50, style_dim) * 0.02)
        else:
            self.register_parameter("style_prototype", None)

    def velocity(self, noisy_latent, text_emb, style_ttl, latent_mask, text_mask, t_norm,
                 style_prototype: torch.Tensor | None = None):
        """Return velocity v_θ(z_t, cond, t) only (for training — no ODE step applied).

        Args:
            noisy_latent : [B, latent_dim, L]   z_t (interpolated noise/data)
            text_emb     : [B, 256, T]          from text_encoder
            style_ttl    : [B, 50, 256]         from style encoder (or cached voice style)
            latent_mask  : [B, 1, L]
            text_mask    : [B, 1, T]
            t_norm       : [B] in [0, 1]        normalized time (=current_step/total_step)
        Returns:
            v : [B, latent_dim, L]  velocity prediction
        """
        if style_prototype is None:
            style_prototype = self.style_prototype
        if style_prototype is None:
            raise ValueError("VectorField.velocity requires style_prototype when learn_style_prototype=False")

        time_emb = self.time_encoder(t_norm)                  # [B, time_dim]
        x = self.proj_in(noisy_latent) * latent_mask
        for blk in self.main_blocks:
            x = blk(x, time_emb, text_emb, style_ttl, latent_mask, text_mask, style_prototype)
        x = self.last_convnext(x, latent_mask)
        v = self.proj_out(x) * latent_mask
        return v

    def forward(self, noisy_latent, text_emb, style_ttl, latent_mask, text_mask,
                current_step, total_step, style_prototype: torch.Tensor | None = None):
        """Inference forward: single Euler ODE step (matches vector_estimator.onnx output)."""
        t_norm = current_step / total_step
        v = self.velocity(
            noisy_latent, text_emb, style_ttl, latent_mask, text_mask, t_norm,
            style_prototype=style_prototype,
        )
        dt = (1.0 / total_step).view(-1, 1, 1)
        denoised = (noisy_latent + dt * v) * latent_mask
        return denoised


# ========================== weight loader ==========================
import onnx, numpy as np
from onnx import numpy_helper


def load_ve_weights(model: VectorField, onnx_path: str):
    m = onnx.load(onnx_path)
    inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}

    def cp(p, arr, name):
        arr = arr.reshape(tuple(p.shape)) if tuple(p.shape) != arr.shape else arr
        with torch.no_grad():
            p.copy_(torch.from_numpy(arr.astype("float32")))

    def cp_linear(linear, mm_name, bias_name):
        W = inits[mm_name].T   # [out, in]
        cp(linear.weight, W, mm_name)
        cp(linear.bias, inits[bias_name], bias_name)

    # proj_in / proj_out
    P = "tts.ttl.vector_field"
    cp(model.proj_in.weight, inits[f"{P}.proj_in.net.weight"], "proj_in")
    cp(model.proj_out.weight, inits[f"{P}.proj_out.net.weight"], "proj_out")

    # time_encoder
    cp(model.time_encoder.mlp[0].weight, inits[f"{P}.time_encoder.mlp.0.linear.weight"], "te.0.w")
    cp(model.time_encoder.mlp[0].bias,   inits[f"{P}.time_encoder.mlp.0.linear.bias"],   "te.0.b")
    cp(model.time_encoder.mlp[2].weight, inits[f"{P}.time_encoder.mlp.2.linear.weight"], "te.2.w")
    cp(model.time_encoder.mlp[2].bias,   inits[f"{P}.time_encoder.mlp.2.linear.bias"],   "te.2.b")

    # MatMul constant mapping per block index (from inspecting ONNX):
    # Each outer block uses 8 onnx::MatMul_* constants arranged sequentially.
    # block 0: 3095 (time FiLM), 3101/2/3 (text attn Q/K/V), 3110 (text attn out), 3116/7/8 (style Q/K/V), 3119 (style out)
    # block 1: 3140, 3146/7/8, 3155, 3161/2/3, 3164
    # block 2: 3185, 3191/2/3, 3200, 3206/7/8, 3209
    # block 3: 3230, 3236/7/8, 3245, 3251/2/3, 3254
    MM = {
        0: {"time": 3095, "text_q": 3101, "text_k": 3102, "text_v": 3103, "text_o": 3110,
            "style_q": 3116, "style_k": 3117, "style_v": 3118, "style_o": 3119},
        1: {"time": 3140, "text_q": 3146, "text_k": 3147, "text_v": 3148, "text_o": 3155,
            "style_q": 3161, "style_k": 3162, "style_v": 3163, "style_o": 3164},
        2: {"time": 3185, "text_q": 3191, "text_k": 3192, "text_v": 3193, "text_o": 3200,
            "style_q": 3206, "style_k": 3207, "style_v": 3208, "style_o": 3209},
        3: {"time": 3230, "text_q": 3236, "text_k": 3237, "text_v": 3238, "text_o": 3245,
            "style_q": 3251, "style_k": 3252, "style_v": 3253, "style_o": 3254},
    }

    def load_convnext(stack, prefix):
        for i, lyr in enumerate(stack.layers):
            pfx = f"{prefix}.convnext.{i}"
            cp(lyr.dwconv.weight, inits[f"{pfx}.dwconv.weight"], f"{pfx}.dw.w")
            cp(lyr.dwconv.bias,   inits[f"{pfx}.dwconv.bias"],   f"{pfx}.dw.b")
            cp(lyr.norm.weight,   inits[f"{pfx}.norm.norm.weight"], f"{pfx}.ln.w")
            cp(lyr.norm.bias,     inits[f"{pfx}.norm.norm.bias"],   f"{pfx}.ln.b")
            cp(lyr.pwconv1.weight, inits[f"{pfx}.pwconv1.weight"], f"{pfx}.pw1.w")
            cp(lyr.pwconv1.bias,   inits[f"{pfx}.pwconv1.bias"],   f"{pfx}.pw1.b")
            cp(lyr.pwconv2.weight, inits[f"{pfx}.pwconv2.weight"], f"{pfx}.pw2.w")
            cp(lyr.pwconv2.bias,   inits[f"{pfx}.pwconv2.bias"],   f"{pfx}.pw2.b")
            cp(lyr.gamma,          inits[f"{pfx}.gamma"],          f"{pfx}.gamma")

    for i, blk in enumerate(model.main_blocks):
        base = 6 * i
        B = MM[i]
        # convnext_0 (4L) at main_blocks.{base+0}
        load_convnext(blk.convnext_0, f"{P}.main_blocks.{base+0}")
        # time FiLM at main_blocks.{base+1}
        cp_linear(blk.time_film.linear, f"onnx::MatMul_{B['time']}", f"{P}.main_blocks.{base+1}.linear.linear.bias")
        # convnext_1 (1L) at main_blocks.{base+2}
        load_convnext(blk.convnext_1, f"{P}.main_blocks.{base+2}")
        # text_attn at main_blocks.{base+3}
        a3 = f"{P}.main_blocks.{base+3}.attn"
        cp_linear(blk.text_attn.W_query, f"onnx::MatMul_{B['text_q']}", f"{a3}.W_query.linear.bias")
        cp_linear(blk.text_attn.W_key,   f"onnx::MatMul_{B['text_k']}", f"{a3}.W_key.linear.bias")
        cp_linear(blk.text_attn.W_value, f"onnx::MatMul_{B['text_v']}", f"{a3}.W_value.linear.bias")
        cp_linear(blk.text_attn.out_fc,  f"onnx::MatMul_{B['text_o']}", f"{a3}.out_fc.linear.bias")
        # theta / increments: ONNX only stores these on main_blocks.3 and shares across blocks
        theta = inits["tts.ttl.vector_field.main_blocks.3.attn.theta"].astype("float32")
        inc   = inits["tts.ttl.vector_field.main_blocks.3.attn.increments"].astype("float32")
        cp(blk.text_attn.theta, theta, "theta")
        cp(blk.text_attn.increments, inc, "increments")
        # LN after text attn: at main_blocks.{base+3}.norm
        cp(blk.text_norm.weight, inits[f"{P}.main_blocks.{base+3}.norm.norm.weight"], f"tn.{i}.w")
        cp(blk.text_norm.bias,   inits[f"{P}.main_blocks.{base+3}.norm.norm.bias"],   f"tn.{i}.b")
        # convnext_2 (1L) at main_blocks.{base+4}
        load_convnext(blk.convnext_2, f"{P}.main_blocks.{base+4}")
        # style_attn at main_blocks.{base+5}
        a5 = f"{P}.main_blocks.{base+5}.attention"
        cp_linear(blk.style_attn.W_query, f"onnx::MatMul_{B['style_q']}", f"{a5}.W_query.linear.bias")
        cp_linear(blk.style_attn.W_key,   f"onnx::MatMul_{B['style_k']}", f"{a5}.W_key.linear.bias")
        cp_linear(blk.style_attn.W_value, f"onnx::MatMul_{B['style_v']}", f"{a5}.W_value.linear.bias")
        cp_linear(blk.style_attn.out_fc,  f"onnx::MatMul_{B['style_o']}", f"{a5}.out_fc.linear.bias")
        # LN after style attn
        cp(blk.style_norm.weight, inits[f"{P}.main_blocks.{base+5}.norm.norm.weight"], f"sn.{i}.w")
        cp(blk.style_norm.bias,   inits[f"{P}.main_blocks.{base+5}.norm.norm.bias"],   f"sn.{i}.b")

    # last_convnext
    load_convnext(model.last_convnext, f"{P}.last_convnext")

    # style prototype (baked into ONNX as initializer /Expand_output_0 with shape [1, 50, 256])
    proto = inits["/Expand_output_0"]
    cp(model.style_prototype, proto, "style_prototype")


# ========================== verification ==========================
def verify(onnx_path: str):
    import onnxruntime as ort
    np.random.seed(0); torch.manual_seed(0)
    B, L, T = 2, 17, 25
    noisy_latent = np.random.randn(B, 144, L).astype(np.float32) * 0.3
    text_emb  = np.random.randn(B, 256, T).astype(np.float32) * 0.3
    style_ttl = np.random.randn(B, 50, 256).astype(np.float32) * 0.3
    latent_mask = np.zeros((B, 1, L), dtype=np.float32); latent_mask[0,0,:15]=1; latent_mask[1,0,:]=1
    text_mask   = np.zeros((B, 1, T), dtype=np.float32); text_mask[0,0,:22]=1; text_mask[1,0,:]=1
    current_step = np.array([1.0, 2.0], dtype=np.float32)
    total_step   = np.array([5.0, 5.0], dtype=np.float32)

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inputs = dict(noisy_latent=noisy_latent, text_emb=text_emb, style_ttl=style_ttl,
                  latent_mask=latent_mask, text_mask=text_mask,
                  current_step=current_step, total_step=total_step)
    y_onnx = sess.run(None, inputs)[0]

    model = VectorField()
    load_ve_weights(model, onnx_path)
    model.eval()
    with torch.no_grad():
        y_torch = model(
            torch.from_numpy(noisy_latent), torch.from_numpy(text_emb), torch.from_numpy(style_ttl),
            torch.from_numpy(latent_mask), torch.from_numpy(text_mask),
            torch.from_numpy(current_step), torch.from_numpy(total_step),
        ).numpy()
    d = np.abs(y_onnx - y_torch)
    return {"shape_onnx": y_onnx.shape, "shape_torch": y_torch.shape,
            "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
            "onnx_range": (float(y_onnx.min()), float(y_onnx.max()))}


if __name__ == "__main__":
    import json
    ONNX = os.path.join(os.path.dirname(__file__), "..", "assets", "onnx", "vector_estimator.onnx")
    print(json.dumps(verify(ONNX), indent=2))
