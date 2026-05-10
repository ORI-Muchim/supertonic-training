"""PyTorch reimplementation of text_encoder.onnx.

Structure:
  text_ids[B,T], style_ttl[B,50,256], text_mask[B,1,T]
   └─ text_embedder (char 163 → 256)
   └─ main stack (masked; symmetric edge pad ksz=5)
       ├─ 6× ConvNeXt(256, inter=1024)
       ├─ 4× AttnEncoder(256, 4 heads head_dim=64, rel-pos window=4, ffn=1024) — skip around the stack
   └─ speech_prompted_text_encoder
       ├─ attention1 (prototype-key cross-attn: K = tanh(learnable[2,128,50]), V = W_value(style))
       │            + residual
       ├─ attention2 (same, different weights + prototype)
       │            + residual
       └─ LayerNorm → * mask
   └─ output text_emb[B,256,T]

Key insight: K is NOT computed from style. K is a fixed learnable prototype tensor shape [H, head_dim, n_style] passed through tanh. Stored as baked-in tanh buffer in ONNX.
"""
from __future__ import annotations
import math, os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
# reuse base blocks
from torch_duration_predictor import (
    MaskedConvNeXt1D, AttnEncoderLayer, RelPosAttention,
    RoPEAttnEncoderLayer,
    symmetric_pad_rep,
)


class TextMainEncoder(nn.Module):
    def __init__(self, vocab=163, dim=256, n_convnext=6, inter_conv=1024,
                 n_attn=4, n_heads=4, ffn_inter=1024, ksz=5, window=4,
                 attn_type: str = "relpos"):
        """attn_type:
            'relpos' — RelPosAttention with windowed learned bias (shipped/ONNX-compat).
            'rope'   — RoPE half-split rotary embeddings (paper A.2.2 line 1075-1076).
        """
        super().__init__()
        self.attn_type = attn_type
        self.char_embedder = nn.Embedding(vocab, dim)
        self.convnext = nn.ModuleList([MaskedConvNeXt1D(dim, inter_conv, ksz, 1) for _ in range(n_convnext)])
        if attn_type == "relpos":
            self.attn_layers = nn.ModuleList([
                AttnEncoderLayer(dim, n_heads, ffn_inter, window_size=window) for _ in range(n_attn)
            ])
        elif attn_type == "rope":
            self.attn_layers = nn.ModuleList([
                RoPEAttnEncoderLayer(dim, n_heads, ffn_inter) for _ in range(n_attn)
            ])
        else:
            raise ValueError(f"unknown attn_type: {attn_type}")

    def forward(self, text_ids, text_mask):
        e = self.char_embedder(text_ids).transpose(1, 2)  # [B, dim, T]
        x = e * text_mask
        for blk in self.convnext:
            x = blk(x, text_mask)
        x_after_conv = x
        for lyr in self.attn_layers:
            x = lyr(x, text_mask)
        x = (x + x_after_conv) * text_mask
        return x


class PrototypeCrossAttention(nn.Module):
    """Cross-attention where K is a fixed learnable prototype (post tanh),
    V = W_value(style), Q = W_query(x). 2 heads × head_dim=128 (for text_encoder).

    style_dim: dim of input style tokens (paper: 128, shipped: 256).
              Defaults to dim for backward compat with shipped weights.
    """
    def __init__(self, dim=256, n_heads=2, n_style=50, style_dim: int | None = None):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.n_style = n_style
        if style_dim is None:
            style_dim = dim
        self.style_dim = style_dim
        # Linear W_query, W_value, out_fc (W_key has no weight — K comes from prototype)
        self.W_query = nn.Linear(dim, dim, bias=True)
        self.W_value = nn.Linear(style_dim, dim, bias=True)   # style_dim → dim (paper: 128 → 256)
        self.out_fc = nn.Linear(dim, dim, bias=True)
        # learnable prototype key: shape stored post-tanh as [n_heads, 1, head_dim, n_style]
        # At training time this would be a pre-tanh parameter; here we keep tanh output as buffer for exact match.
        self.register_buffer("k_tanh", torch.zeros(n_heads, 1, self.head_dim, n_style))

    def forward(self, x, style, text_mask):
        """x: [B, T, C]; style: [B, n_style, C]; text_mask: [B, 1, T]."""
        B, T, C = x.shape
        q = self.W_query(x)               # [B, T, C]
        v = self.W_value(style)           # [B, n_style, C]
        # split heads (via Split+Unsqueeze+Concat as per ONNX: axis=-1 split into n_heads equal parts, then stack on dim=1)
        q = q.reshape(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)   # [B, H, T, D]
        v = v.reshape(B, self.n_style, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, 50, D]

        # K is fixed prototype (same across batches)
        k = self.k_tanh                     # [H, 1, D, n_style]
        # scores = Q @ K : [B,H,T,D] @ [H,1,D,n_style] -> [B,H,T,n_style]
        scores = torch.matmul(q, k.squeeze(1))  # squeeze to [H, D, n_style]; broadcast over B
        # ONNX uses sqrt(dim) (not sqrt(head_dim)): observed Div by 16.0 = sqrt(256)
        scores = scores / math.sqrt(self.dim)
        # softmax over keys (n_style)
        attn = F.softmax(scores, dim=-1)
        # zero out attention for masked queries (ONNX: Where after Softmax)
        mask_q = text_mask.unsqueeze(1)     # [B, 1, 1, T]
        mask_q = mask_q.transpose(-1, -2)   # [B, 1, T, 1]
        attn = attn * mask_q
        # context = attn @ V : [B,H,T,n_style] @ [B,H,n_style,D] -> [B,H,T,D]
        ctx = torch.matmul(attn, v)
        # merge heads: [B,H,T,D] -> [B,T,H*D]
        ctx = ctx.permute(0, 2, 1, 3).reshape(B, T, C)
        out = self.out_fc(ctx)
        # Apply mask (ONNX: Mul after out_fc)
        out = out * text_mask.transpose(-1, -2)
        return out


class LearnablePrototypeCrossAttention(nn.Module):
    """Paper-faithful cross-attention for from-scratch TTL training.

    Q comes from text states, K comes from the shared 50x128 reference-key
    prototype, and V comes from reference-value style tokens. Unlike the ONNX
    compatibility path above, the prototype is trainable and projected through
    W_key at runtime.
    """
    def __init__(self, dim=128, n_heads=2, n_style=50, style_dim: int | None = None):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.n_style = n_style
        if style_dim is None:
            style_dim = dim
        self.W_query = nn.Linear(dim, dim, bias=True)
        self.W_key = nn.Linear(dim, dim, bias=True)
        self.W_value = nn.Linear(style_dim, dim, bias=True)
        self.out_fc = nn.Linear(dim, dim, bias=True)

    def forward(self, x, style, text_mask, reference_key):
        """x [B,T,C], style [B,50,style_dim], reference_key [1,50,C]."""
        B, T, C = x.shape
        proto = reference_key.expand(B, -1, -1)
        q = self.W_query(x)
        k = torch.tanh(self.W_key(proto))
        v = self.W_value(style)

        q = q.reshape(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, self.n_style, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, self.n_style, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dim)
        attn = F.softmax(scores, dim=-1)
        mask_q = text_mask.unsqueeze(1).transpose(-1, -2)
        attn = attn * mask_q

        ctx = torch.matmul(attn, v)
        ctx = ctx.permute(0, 2, 1, 3).reshape(B, T, C)
        out = self.out_fc(ctx)
        return out * text_mask.transpose(-1, -2)


class SpeechPromptedTextEncoder(nn.Module):
    def __init__(self, dim=256, n_heads=2, n_style=50, style_dim: int | None = None):
        super().__init__()
        self.attention1 = PrototypeCrossAttention(dim, n_heads, n_style, style_dim=style_dim)
        self.attention2 = PrototypeCrossAttention(dim, n_heads, n_style, style_dim=style_dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x_in_conv, style, text_mask):
        """x_in_conv: [B, 256, T] (output of main text encoder). Returns [B, 256, T].

        Residual structure (from ONNX Add_1 = attn2_out + Transpose_output (= x_orig)):
          x_orig = x_in_conv.T
          q1 = x_orig + attn1(x_orig)    -> used only as Q input for attn2
          out = x_orig + attn2(q1)        -> final residual uses ORIGINAL x, not q1
        """
        x_orig = x_in_conv.transpose(1, 2)                # [B, T, 256]
        q1 = x_orig + self.attention1(x_orig, style, text_mask)
        x = x_orig + self.attention2(q1, style, text_mask)
        x = self.norm(x)
        x = x.transpose(1, 2)                              # back to [B, 256, T]
        x = x * text_mask
        return x


class SpeechPromptedTextEncoderPaper(nn.Module):
    """Paper-faithful speech-prompted text encoder.

    The shared `reference_key` is the 50 learnable vectors described in A.2.2;
    train_ttl also passes the same parameter into the VF estimator.
    """
    def __init__(self, dim=128, n_heads=2, n_style=50, style_dim: int | None = None):
        super().__init__()
        self.attention1 = LearnablePrototypeCrossAttention(dim, n_heads, n_style, style_dim=style_dim)
        self.attention2 = LearnablePrototypeCrossAttention(dim, n_heads, n_style, style_dim=style_dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x_in_conv, style, text_mask, reference_key):
        x_orig = x_in_conv.transpose(1, 2)
        q1 = x_orig + self.attention1(x_orig, style, text_mask, reference_key)
        x = x_orig + self.attention2(q1, style, text_mask, reference_key)
        x = self.norm(x)
        x = x.transpose(1, 2)
        return x * text_mask


class TextEncoder(nn.Module):
    def __init__(self, vocab=163, dim=256, n_style=50, style_dim: int | None = None):
        super().__init__()
        self.text_encoder = TextMainEncoder(vocab=vocab, dim=dim)
        self.speech_prompted_text_encoder = SpeechPromptedTextEncoder(
            dim=dim, n_heads=2, n_style=n_style, style_dim=style_dim,
        )

    def forward(self, text_ids, style_ttl, text_mask):
        x = self.text_encoder(text_ids, text_mask)                          # [B, 256, T]
        x = self.speech_prompted_text_encoder(x, style_ttl, text_mask)      # [B, 256, T]
        return x


class TextEncoderPaper(nn.Module):
    """Paper-faithful TTL text encoder for from-scratch training.

    Paper A.2.2:
      - character embedding dim = 128
      - 6 ConvNeXt blocks, kernel=5, intermediate=512
      - 4 self-attention blocks, 4 heads, FFN/filter channels=512, RoPE
      - 2 cross-attention layers
      - 50 learnable reference-key vectors, dim=128, reused by the VF estimator

    attn_type='rope' is the paper-faithful default. Use 'relpos' for the legacy
    rel-pos-window=4 path (closer to the released ONNX, but not what the paper
    text specifies).
    """
    def __init__(self, vocab=163, dim=128, n_style=50, style_dim: int = 128,
                 attn_type: str = "rope"):
        super().__init__()
        self.reference_key = nn.Parameter(torch.randn(1, n_style, dim) * 0.02)
        self.text_encoder = TextMainEncoder(
            vocab=vocab,
            dim=dim,
            n_convnext=6,
            inter_conv=512,
            n_attn=4,
            n_heads=4,
            ffn_inter=512,
            ksz=5,
            window=4,
            attn_type=attn_type,
        )
        self.speech_prompted_text_encoder = SpeechPromptedTextEncoderPaper(
            dim=dim, n_heads=2, n_style=n_style, style_dim=style_dim,
        )

    def forward(self, text_ids, style_ttl, text_mask):
        x = self.text_encoder(text_ids, text_mask)
        x = self.speech_prompted_text_encoder(x, style_ttl, text_mask, self.reference_key)
        return x


# ===== weight loader ==============================
import onnx, numpy as np
from onnx import numpy_helper


def load_text_encoder_weights(model: TextEncoder, onnx_path: str):
    m = onnx.load(onnx_path)
    inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}

    def cp(p, a, name):
        a = a.reshape(tuple(p.shape)) if tuple(p.shape) != a.shape else a
        with torch.no_grad():
            p.copy_(torch.from_numpy(a.astype("float32")))

    # --- main text encoder ---
    te = model.text_encoder
    P = "tts.ttl.text_encoder"
    cp(te.char_embedder.weight, inits[f"{P}.text_embedder.char_embedder.weight"], "char")
    for i, blk in enumerate(te.convnext):
        pfx = f"{P}.convnext.convnext.{i}"
        cp(blk.dwconv.weight, inits[f"{pfx}.dwconv.weight"], f"c{i}.dw.w")
        cp(blk.dwconv.bias,   inits[f"{pfx}.dwconv.bias"],   f"c{i}.dw.b")
        cp(blk.norm.weight,   inits[f"{pfx}.norm.norm.weight"], f"c{i}.ln.w")
        cp(blk.norm.bias,     inits[f"{pfx}.norm.norm.bias"],   f"c{i}.ln.b")
        cp(blk.pwconv1.weight, inits[f"{pfx}.pwconv1.weight"], f"c{i}.pw1.w")
        cp(blk.pwconv1.bias,   inits[f"{pfx}.pwconv1.bias"],   f"c{i}.pw1.b")
        cp(blk.pwconv2.weight, inits[f"{pfx}.pwconv2.weight"], f"c{i}.pw2.w")
        cp(blk.pwconv2.bias,   inits[f"{pfx}.pwconv2.bias"],   f"c{i}.pw2.b")
        cp(blk.gamma,          inits[f"{pfx}.gamma"],          f"c{i}.gamma")
    for i, lyr in enumerate(te.attn_layers):
        pfx = f"{P}.attn_encoder"
        cp(lyr.attn.conv_q.weight, inits[f"{pfx}.attn_layers.{i}.conv_q.weight"], f"a{i}.q.w")
        cp(lyr.attn.conv_q.bias,   inits[f"{pfx}.attn_layers.{i}.conv_q.bias"],   f"a{i}.q.b")
        cp(lyr.attn.conv_k.weight, inits[f"{pfx}.attn_layers.{i}.conv_k.weight"], f"a{i}.k.w")
        cp(lyr.attn.conv_k.bias,   inits[f"{pfx}.attn_layers.{i}.conv_k.bias"],   f"a{i}.k.b")
        cp(lyr.attn.conv_v.weight, inits[f"{pfx}.attn_layers.{i}.conv_v.weight"], f"a{i}.v.w")
        cp(lyr.attn.conv_v.bias,   inits[f"{pfx}.attn_layers.{i}.conv_v.bias"],   f"a{i}.v.b")
        cp(lyr.attn.conv_o.weight, inits[f"{pfx}.attn_layers.{i}.conv_o.weight"], f"a{i}.o.w")
        cp(lyr.attn.conv_o.bias,   inits[f"{pfx}.attn_layers.{i}.conv_o.bias"],   f"a{i}.o.b")
        cp(lyr.attn.emb_rel_k,     inits[f"{pfx}.attn_layers.{i}.emb_rel_k"],     f"a{i}.rk")
        cp(lyr.attn.emb_rel_v,     inits[f"{pfx}.attn_layers.{i}.emb_rel_v"],     f"a{i}.rv")
        cp(lyr.norm1.weight,       inits[f"{pfx}.norm_layers_1.{i}.norm.weight"], f"a{i}.n1.w")
        cp(lyr.norm1.bias,         inits[f"{pfx}.norm_layers_1.{i}.norm.bias"],   f"a{i}.n1.b")
        cp(lyr.conv1.weight,       inits[f"{pfx}.ffn_layers.{i}.conv_1.weight"],  f"a{i}.f1.w")
        cp(lyr.conv1.bias,         inits[f"{pfx}.ffn_layers.{i}.conv_1.bias"],    f"a{i}.f1.b")
        cp(lyr.conv2.weight,       inits[f"{pfx}.ffn_layers.{i}.conv_2.weight"],  f"a{i}.f2.w")
        cp(lyr.conv2.bias,         inits[f"{pfx}.ffn_layers.{i}.conv_2.bias"],    f"a{i}.f2.b")
        cp(lyr.norm2.weight,       inits[f"{pfx}.norm_layers_2.{i}.norm.weight"], f"a{i}.n2.w")
        cp(lyr.norm2.bias,         inits[f"{pfx}.norm_layers_2.{i}.norm.bias"],   f"a{i}.n2.b")
    # (no proj_out in text_encoder — unlike DP)

    # --- speech_prompted_text_encoder ---
    sp = model.speech_prompted_text_encoder
    SP = "tts.ttl.speech_prompted_text_encoder"
    # MatMul constants: need to identify which is which by position usage. From inspection:
    #   3678 = attention1.W_query.weight (256, 256)
    #   3680 = attention1.W_value.weight
    #   3681 = attention1.out_fc.weight
    #   3682 = attention2.W_query.weight
    #   3684 = attention2.W_value.weight
    #   3685 = attention2.out_fc.weight
    # Linear.weight shape is [out, in]; ONNX MatMul is [in, out] -> transpose.
    def cp_linear(linear, mm_name, bias_name):
        W = inits[mm_name].T    # [out, in]
        cp(linear.weight, W, mm_name)
        cp(linear.bias, inits[bias_name], bias_name)

    cp_linear(sp.attention1.W_query, "onnx::MatMul_3678", f"{SP}.attention1.W_query.linear.bias")
    cp_linear(sp.attention1.W_value, "onnx::MatMul_3680", f"{SP}.attention1.W_value.linear.bias")
    cp_linear(sp.attention1.out_fc,  "onnx::MatMul_3681", f"{SP}.attention1.out_fc.linear.bias")
    cp_linear(sp.attention2.W_query, "onnx::MatMul_3682", f"{SP}.attention2.W_query.linear.bias")
    cp_linear(sp.attention2.W_value, "onnx::MatMul_3684", f"{SP}.attention2.W_value.linear.bias")
    cp_linear(sp.attention2.out_fc,  "onnx::MatMul_3685", f"{SP}.attention2.out_fc.linear.bias")

    # tanh prototype keys (precomputed in ONNX graph)
    k1 = inits["/speech_prompted_text_encoder/attention1/tanh/Tanh_output_0"]
    k2 = inits["/speech_prompted_text_encoder/attention2/tanh/Tanh_output_0"]
    # shape: (2, 1, 128, 50) -> want to store as [H, 1, D, 50] ✓
    with torch.no_grad():
        sp.attention1.k_tanh.copy_(torch.from_numpy(k1.astype("float32")))
        sp.attention2.k_tanh.copy_(torch.from_numpy(k2.astype("float32")))

    cp(sp.norm.weight, inits[f"{SP}.norm.norm.weight"], "sp.ln.w")
    cp(sp.norm.bias,   inits[f"{SP}.norm.norm.bias"],   "sp.ln.b")


def verify(onnx_path: str):
    import onnxruntime as ort
    np.random.seed(0); torch.manual_seed(0)
    B, T = 2, 35
    text_ids = np.random.randint(0, 162, size=(B, T)).astype(np.int64)
    style_ttl = np.random.randn(B, 50, 256).astype(np.float32) * 0.3
    text_mask = np.zeros((B, 1, T), dtype=np.float32)
    text_mask[0, 0, :30] = 1.0; text_mask[1, 0, :T] = 1.0

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    y_onnx = sess.run(None, {"text_ids": text_ids, "style_ttl": style_ttl, "text_mask": text_mask})[0]

    model = TextEncoder()
    load_text_encoder_weights(model, onnx_path)
    model.eval()
    with torch.no_grad():
        y_torch = model(torch.from_numpy(text_ids), torch.from_numpy(style_ttl), torch.from_numpy(text_mask)).numpy()
    d = np.abs(y_onnx - y_torch)
    return {"shape_onnx": y_onnx.shape, "shape_torch": y_torch.shape,
            "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean())}


if __name__ == "__main__":
    import os, json
    ONNX = os.path.join(os.path.dirname(__file__), "..", "assets", "onnx", "text_encoder.onnx")
    print(json.dumps(verify(ONNX), indent=2))
