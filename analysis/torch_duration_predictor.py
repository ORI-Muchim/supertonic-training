"""PyTorch reimplementation of duration_predictor.onnx.

Structure (reverse-engineered):
  text_ids[B,T], style_dp[B,8,16], text_mask[B,1,T]
   └─ sentence_encoder
       ├─ char_embedder (163, 64)
       ├─ prepend learnable sentence_token → [B, 64, T+1]
       ├─ 6× masked ConvNeXt (hdim=64, inter=256, ksz=5, causal REPLICATE pad)
       ├─ 2× masked Attention encoder (rel-pos, 2 heads, head_dim=32, window=4, pre-norm)
       │    residual skip around the whole stack
       ├─ proj_out Conv1d 64→64 ksz=1
       └─ take sentence_token position [:, :, :1] → [B, 64, 1]
   └─ predictor
       ├─ flatten style_dp [B,8,16] → [B, 128]; flatten sentence_vec → [B, 64]
       ├─ concat → [B, 192]  (sentence first)
       ├─ Linear 192→128 → PReLU(1) → Linear 128→1 → Exp
       └─ duration [B]
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_pad_rep(x, ksz, dilation=1):
    return F.pad(x, ((ksz - 1) * dilation, 0), mode="replicate")


def symmetric_pad_rep(x, ksz, dilation=1):
    pad = (ksz - 1) * dilation // 2
    return F.pad(x, (pad, pad), mode="replicate")


# ----- ConvNeXt with input/output mask application ----------
class MaskedConvNeXt1D(nn.Module):
    def __init__(self, dim, inter, ksz, dilation):
        super().__init__()
        self.ksz, self.dilation = ksz, dilation
        self.dwconv = nn.Conv1d(dim, dim, ksz, groups=dim, dilation=dilation, bias=True)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Conv1d(dim, inter, 1, bias=True)
        self.pwconv2 = nn.Conv1d(inter, dim, 1, bias=True)
        self.gamma = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x, mask):
        # mask: [B, 1, T]
        x = x * mask                                      # input mask
        dw = self.dwconv(symmetric_pad_rep(x, self.ksz, self.dilation))
        dw = dw * mask                                    # mask after dwconv
        h = self.norm(dw.transpose(1, 2)).transpose(1, 2)
        h = self.pwconv1(h)
        h = F.gelu(h, approximate="none")
        h = self.pwconv2(h)
        h = self.gamma * h
        h = h * mask                                      # mask gamma*pw2
        out = x + h                                       # residual
        out = out * mask                                  # output mask
        return out


# ----- VITS/Glow-TTS style relative-position self-attention -----
class RelPosAttention(nn.Module):
    """Input [B, C, T]. Window_size=4 (2W+1=9). Channels must be divisible by n_heads."""
    def __init__(self, channels, n_heads, window_size=4):
        super().__init__()
        assert channels % n_heads == 0
        self.channels = channels
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.window_size = window_size
        self.conv_q = nn.Conv1d(channels, channels, 1, bias=True)
        self.conv_k = nn.Conv1d(channels, channels, 1, bias=True)
        self.conv_v = nn.Conv1d(channels, channels, 1, bias=True)
        self.conv_o = nn.Conv1d(channels, channels, 1, bias=True)
        # relative position embeddings shared across heads
        rel_stddev = self.head_dim ** -0.5
        self.emb_rel_k = nn.Parameter(torch.randn(1, 2 * window_size + 1, self.head_dim) * rel_stddev)
        self.emb_rel_v = nn.Parameter(torch.randn(1, 2 * window_size + 1, self.head_dim) * rel_stddev)

    @staticmethod
    def _relative_position_to_absolute(x):
        """x: [B, H, L, 2L-1] -> [B, H, L, L] via direct gather.
        abs[i, j] = rel[i, j - i + (L - 1)]"""
        B, H, L, R = x.shape
        assert R == 2 * L - 1
        i = torch.arange(L, device=x.device).unsqueeze(1)     # [L, 1]
        j = torch.arange(L, device=x.device).unsqueeze(0)     # [1, L]
        idx = (j - i + (L - 1)).long()                         # [L, L]
        idx_e = idx.view(1, 1, L, L).expand(B, H, L, L)
        return x.gather(-1, idx_e)

    @staticmethod
    def _absolute_position_to_relative(x):
        """Inverse: [B, H, L, L] -> [B, H, L, 2L-1]. rel[i, r] = abs[i, r + i - (L-1)] if in range else 0."""
        B, H, L, _ = x.shape
        out = x.new_zeros(B, H, L, 2 * L - 1)
        i = torch.arange(L, device=x.device).unsqueeze(1)      # [L, 1]
        r = torch.arange(2 * L - 1, device=x.device).unsqueeze(0)  # [1, 2L-1]
        j = r + i - (L - 1)                                     # [L, 2L-1]
        valid = (j >= 0) & (j < L)
        j_clamped = j.clamp(0, L - 1)
        idx_e = j_clamped.view(1, 1, L, 2 * L - 1).expand(B, H, L, 2 * L - 1)
        gathered = x.gather(-1, idx_e)
        mask = valid.view(1, 1, L, 2 * L - 1).to(x.dtype)
        return gathered * mask

    def _get_relative_embeddings(self, emb, L):
        """emb: [1, 2W+1, D]. Pad/crop to [1, 2L-1, D]."""
        W = self.window_size
        pad_len = max(L - (W + 1), 0)
        slice_start = max((W + 1) - L, 0)
        slice_end = slice_start + (2 * L - 1)
        if pad_len > 0:
            emb = F.pad(emb, (0, 0, pad_len, pad_len))
        return emb[:, slice_start:slice_end]

    def forward(self, x, mask_attn):
        """x:[B,C,T]. mask_attn: [B,1,T,T] attention mask (1=keep, 0=pad)."""
        B, C, T = x.shape
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)
        # reshape to [B, H, D, T] then [B, H, T, D]
        q = q.reshape(B, self.n_heads, self.head_dim, T).transpose(-1, -2)  # [B,H,T,D]
        k = k.reshape(B, self.n_heads, self.head_dim, T).transpose(-1, -2)
        v = v.reshape(B, self.n_heads, self.head_dim, T).transpose(-1, -2)
        # content score
        scores = torch.matmul(q / math.sqrt(self.head_dim), k.transpose(-1, -2))  # [B,H,T,T]
        # relative key contribution
        rel_k = self._get_relative_embeddings(self.emb_rel_k, T)              # [1, 2T-1, D]
        scores_local = torch.matmul(q / math.sqrt(self.head_dim), rel_k.transpose(-1, -2).unsqueeze(0))  # [B,H,T,2T-1]
        scores_local = self._relative_position_to_absolute(scores_local)                  # [B,H,T,T]
        scores = scores + scores_local
        # mask
        scores = scores.masked_fill(mask_attn == 0, -1e4)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # [B,H,T,D]
        # relative value contribution
        rel_weight = self._absolute_position_to_relative(attn)                # [B,H,T,2T-1]
        rel_v = self._get_relative_embeddings(self.emb_rel_v, T).unsqueeze(0) # [1,1,2T-1,D]
        out = out + torch.matmul(rel_weight, rel_v.squeeze(0))
        # back to [B, C, T]
        out = out.transpose(-1, -2).reshape(B, C, T)
        return self.conv_o(out)


class AttnEncoderLayer(nn.Module):
    def __init__(self, dim, n_heads, ffn_inter, window_size=4):
        super().__init__()
        self.attn = RelPosAttention(dim, n_heads, window_size)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.conv1 = nn.Conv1d(dim, ffn_inter, 1, bias=True)
        self.conv2 = nn.Conv1d(ffn_inter, dim, 1, bias=True)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x, mask):
        # ONNX traces show: attn, add residual x (pre-attn), then LN1. Then FFN, add residual (post-LN1), LN2.
        # Layout: x -> attn -> +x -> LN1 -> ffn -> +LN1 -> LN2
        mask_attn = mask.unsqueeze(2) * mask.unsqueeze(3)  # [B,1,T,T]
        a = self.attn(x, mask_attn)
        a = a * mask
        x = x + a
        # LN1 expects channels-last
        x = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        x = x * mask
        f = self.conv1(x)
        f = F.relu(f)
        f = f * mask
        f = self.conv2(f)
        f = f * mask
        x = x + f
        x = self.norm2(x.transpose(1, 2)).transpose(1, 2)
        x = x * mask
        return x


class SentenceEncoder(nn.Module):
    def __init__(self, vocab=163, char_dim=64, n_convnext=6, inter_conv=256,
                 n_attn=2, n_heads=2, ffn_inter=256, ksz=5):
        super().__init__()
        self.char_embedder = nn.Embedding(vocab, char_dim)
        self.sentence_token = nn.Parameter(torch.zeros(1, char_dim, 1))
        self.convnext = nn.ModuleList([MaskedConvNeXt1D(char_dim, inter_conv, ksz, 1) for _ in range(n_convnext)])
        self.attn_layers = nn.ModuleList([AttnEncoderLayer(char_dim, n_heads, ffn_inter) for _ in range(n_attn)])
        self.proj_out = nn.Conv1d(char_dim, char_dim, 1, bias=False)

    def forward(self, text_ids, text_mask):
        B, T = text_ids.shape
        # embed
        e = self.char_embedder(text_ids).transpose(1, 2)       # [B, 64, T]
        e = e * text_mask                                       # mask pads
        # prepend sentence_token (mask is extended with 1 at the front)
        st = self.sentence_token.expand(B, -1, 1)
        x = torch.cat([st, e], dim=-1)                          # [B, 64, T+1]
        mask = torch.cat([torch.ones(B, 1, 1, device=x.device, dtype=text_mask.dtype), text_mask], dim=-1)
        # residual around the whole convnext+attn stack: ONNX shows Add(attn_encoder_out, convnext.5_out*mask)
        x_in_convnext = x
        for blk in self.convnext:
            x = blk(x, mask)
        x_after_convnext = x
        for lyr in self.attn_layers:
            x = lyr(x, mask)
        x = x + x_after_convnext                                # skip over attn_encoder
        # take sentence_token position (first)
        sent = x[:, :, :1]                                      # [B, 64, 1]
        mask1 = mask[:, :, :1]
        sent = self.proj_out(sent) * mask1
        return sent


class DurationPredictor(nn.Module):
    def __init__(self, sentence_dim=64, n_style=8, style_dim=16, hdim=128):
        super().__init__()
        self.sentence_encoder = SentenceEncoder(char_dim=sentence_dim)
        self.fc1 = nn.Linear(sentence_dim + n_style * style_dim, hdim)
        self.act = nn.PReLU(num_parameters=1)
        self.fc2 = nn.Linear(hdim, 1)
        self.n_style = n_style
        self.style_dim = style_dim

    def forward(self, text_ids, style_dp, text_mask):
        sent = self.sentence_encoder(text_ids, text_mask)         # [B, 64, 1]
        sent = sent.reshape(sent.shape[0], -1)                    # [B, 64]
        style = style_dp.reshape(style_dp.shape[0], -1)           # [B, 128]
        h = torch.cat([sent, style], dim=-1)                      # [B, 192]
        h = self.fc1(h)
        h = self.act(h)
        h = self.fc2(h)
        dur = torch.exp(h).squeeze(-1)                             # [B]
        return dur


# =============== weight loader ================================
import onnx, numpy as np
from onnx import numpy_helper


def load_dp_weights(model: DurationPredictor, onnx_path: str):
    inits = {i.name: numpy_helper.to_array(i) for i in onnx.load(onnx_path).graph.initializer}
    def cp(p, a, n):
        a = a if a.shape == tuple(p.shape) else a.reshape(p.shape)
        with torch.no_grad():
            p.copy_(torch.from_numpy(a.astype("float32")))

    se = model.sentence_encoder
    P = "tts.dp.sentence_encoder"
    cp(se.char_embedder.weight, inits[f"{P}.text_embedder.char_embedder.weight"], "char_emb")
    cp(se.sentence_token,        inits[f"{P}.sentence_token"], "sent_tok")
    for i, blk in enumerate(se.convnext):
        pfx = f"{P}.convnext.convnext.{i}"
        cp(blk.dwconv.weight, inits[f"{pfx}.dwconv.weight"], f"cn{i}.dw.w")
        cp(blk.dwconv.bias,   inits[f"{pfx}.dwconv.bias"],   f"cn{i}.dw.b")
        cp(blk.norm.weight,   inits[f"{pfx}.norm.norm.weight"], f"cn{i}.ln.w")
        cp(blk.norm.bias,     inits[f"{pfx}.norm.norm.bias"],   f"cn{i}.ln.b")
        cp(blk.pwconv1.weight, inits[f"{pfx}.pwconv1.weight"], f"cn{i}.pw1.w")
        cp(blk.pwconv1.bias,   inits[f"{pfx}.pwconv1.bias"],   f"cn{i}.pw1.b")
        cp(blk.pwconv2.weight, inits[f"{pfx}.pwconv2.weight"], f"cn{i}.pw2.w")
        cp(blk.pwconv2.bias,   inits[f"{pfx}.pwconv2.bias"],   f"cn{i}.pw2.b")
        cp(blk.gamma,          inits[f"{pfx}.gamma"],          f"cn{i}.gamma")
    for i, lyr in enumerate(se.attn_layers):
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
    cp(se.proj_out.weight, inits[f"{P}.proj_out.net.weight"], "proj_out.w")

    cp(model.fc1.weight, inits["tts.dp.predictor.layers.0.weight"], "fc1.w")
    cp(model.fc1.bias,   inits["tts.dp.predictor.layers.0.bias"],   "fc1.b")
    cp(model.act.weight, inits["tts.dp.predictor.activation.weight"].reshape(-1), "act.w")
    cp(model.fc2.weight, inits["tts.dp.predictor.layers.1.weight"], "fc2.w")
    cp(model.fc2.bias,   inits["tts.dp.predictor.layers.1.bias"],   "fc2.b")


def verify(onnx_path: str):
    import onnxruntime as ort
    np.random.seed(0); torch.manual_seed(0)
    B, T = 2, 30
    text_ids = np.random.randint(0, 162, size=(B, T)).astype(np.int64)
    style_dp = np.random.randn(B, 8, 16).astype(np.float32) * 0.2
    text_mask = np.zeros((B, 1, T), dtype=np.float32)
    text_mask[0, 0, :25] = 1.0
    text_mask[1, 0, :T] = 1.0

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    y_onnx = sess.run(None, {"text_ids": text_ids, "style_dp": style_dp, "text_mask": text_mask})[0]

    model = DurationPredictor()
    load_dp_weights(model, onnx_path)
    model.eval()
    with torch.no_grad():
        y_torch = model(torch.from_numpy(text_ids), torch.from_numpy(style_dp), torch.from_numpy(text_mask)).numpy()
    return {
        "onnx": y_onnx.tolist(), "torch": y_torch.tolist(),
        "max_abs_diff": float(np.abs(y_onnx - y_torch).max()),
    }


if __name__ == "__main__":
    import os, json
    ONNX = os.path.join(os.path.dirname(__file__), "..", "assets", "onnx", "duration_predictor.onnx")
    r = verify(ONNX)
    print(json.dumps(r, indent=2))
