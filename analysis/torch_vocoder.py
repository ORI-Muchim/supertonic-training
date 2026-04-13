"""PyTorch reimplementation of vocoder.onnx (= TTL un-chunker + AE decoder).

Reverse-engineered from ONNX graph + initializer names.

Verified structure:
  latent [B, 144, L]
    ÷ normalizer.scale (= 0.25)          ×4 effectively
    reshape [B, 24, 6, L] → transpose [0,1,3,2] → [B, 24, L, 6] → flatten → [B, 24, 6L]
    * latent_std + latent_mean           (de-normalize to AE latent scale)
    stem:  causal Conv1d 24→512, ksz=7
    10× ConvNeXt(hdim=512, inter=2048, ksz=7, dilations=[1,2,4,1,2,4,1,1,1,1]), CAUSAL
    final_norm: BatchNorm1d(512)
    head.layer1:  causal Conv1d 512→2048, ksz=3
    PReLU (1 shared channel)
    head.layer2:  Conv1d 2048→512, ksz=1, NO bias
    transpose [0,2,1] + reshape → waveform [B, 6L·512]
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_pad(x: torch.Tensor, ksz: int, dilation: int) -> torch.Tensor:
    # Supertonic uses REPLICATE (edge) padding — verified by reading /decoder/embed/Pad_output_0
    return F.pad(x, ((ksz - 1) * dilation, 0), mode="replicate")


class ConvNeXt1D(nn.Module):
    """Causal ConvNeXt-1D block used throughout Supertonic."""
    def __init__(self, dim: int, intermediate: int, ksz: int, dilation: int):
        super().__init__()
        self.ksz = ksz
        self.dilation = dilation
        self.dwconv = nn.Conv1d(dim, dim, ksz, groups=dim, dilation=dilation, bias=True)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Conv1d(dim, intermediate, 1, bias=True)
        self.pwconv2 = nn.Conv1d(intermediate, dim, 1, bias=True)
        self.gamma = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = causal_pad(x, self.ksz, self.dilation)
        x = self.dwconv(x)
        # LayerNorm on channel dim: NCL -> NLC -> NCL
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.pwconv1(x)
        x = F.gelu(x, approximate="none")  # exact (Erf) GELU — matches ONNX
        x = self.pwconv2(x)
        x = self.gamma * x
        return residual + x


class Vocoder(nn.Module):
    def __init__(
        self,
        ldim: int = 24,
        chunk_compress_factor: int = 6,
        hdim: int = 512,
        intermediate: int = 2048,
        ksz_init: int = 7,
        ksz: int = 7,
        num_layers: int = 10,
        dilations: tuple[int, ...] = (1, 2, 4, 1, 2, 4, 1, 1, 1, 1),
        head_inter: int = 2048,
        head_out: int = 512,
        head_ksz: int = 3,
        ttl_normalizer_scale: float = 0.25,
    ):
        super().__init__()
        assert len(dilations) == num_layers
        self.ldim = ldim
        self.kc = chunk_compress_factor
        self.hdim = hdim

        # buffers filled from ONNX
        self.register_buffer("normalizer_scale", torch.tensor(ttl_normalizer_scale))
        self.register_buffer("latent_mean", torch.zeros(1, ldim, 1))
        self.register_buffer("latent_std", torch.ones(1, ldim, 1))

        self.stem_ksz = ksz_init
        self.stem = nn.Conv1d(ldim, hdim, ksz_init, bias=True)
        self.convnext = nn.ModuleList(
            [ConvNeXt1D(hdim, intermediate, ksz, d) for d in dilations]
        )
        self.final_norm = nn.BatchNorm1d(hdim)  # matches final_norm.norm.{weight,bias,running_mean,running_var}
        # Head
        self.head_ksz = head_ksz
        self.head_layer1 = nn.Conv1d(hdim, head_inter, head_ksz, bias=True)
        self.head_act = nn.PReLU(num_parameters=1)
        self.head_layer2 = nn.Conv1d(head_inter, head_out, 1, bias=False)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """latent: [B, ldim*kc, L] -> wav: [B, L*kc*head_out]"""
        B = latent.shape[0]
        L = latent.shape[-1]

        # 1) un-normalize by ttl.normalizer.scale (divide, matching onnx /Div)
        x = latent / self.normalizer_scale

        # 2) un-chunk: [B, 144, L] -> [B, 24, 6, L] -> [B, 24, L, 6] -> [B, 24, L*6]
        x = x.reshape(B, self.ldim, self.kc, L)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.reshape(B, self.ldim, L * self.kc)

        # 3) de-normalize AE latent
        x = x * self.latent_std + self.latent_mean

        # 4) stem
        x = causal_pad(x, self.stem_ksz, 1)
        x = self.stem(x)

        # 5) ConvNeXt stack
        for blk in self.convnext:
            x = blk(x)

        # 6) final BN
        x = self.final_norm(x)

        # 7) head
        x = causal_pad(x, self.head_ksz, 1)
        x = self.head_layer1(x)
        x = self.head_act(x)
        x = self.head_layer2(x)  # [B, 512, L*kc]

        # 8) reshape to waveform [B, L*kc*512]
        wav = x.transpose(1, 2).reshape(B, -1)
        return wav


# ===== ONNX -> state_dict loader ===================================

import onnx
from onnx import numpy_helper
import numpy as np


def load_onnx_initializers(path: str) -> dict[str, np.ndarray]:
    m = onnx.load(path)
    return {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}


def load_vocoder_weights(model: Vocoder, onnx_path: str) -> list[str]:
    """Copy weights from vocoder.onnx into a Vocoder module. Returns list of warnings."""
    inits = load_onnx_initializers(onnx_path)
    warnings: list[str] = []

    def take(name: str) -> np.ndarray:
        if name not in inits:
            raise KeyError(name)
        return inits[name]

    def copy(param: torch.Tensor, arr: np.ndarray, tname: str):
        if tuple(param.shape) != tuple(arr.shape):
            raise ValueError(f"shape mismatch on {tname}: {tuple(param.shape)} vs {arr.shape}")
        with torch.no_grad():
            param.copy_(torch.from_numpy(arr.astype("float32")))

    # normalizer + latent stats
    copy(model.normalizer_scale, take("tts.ttl.normalizer.scale"), "normalizer_scale")
    copy(model.latent_mean, take("tts.ae.latent_mean"), "latent_mean")
    copy(model.latent_std, take("tts.ae.latent_std"), "latent_std")

    # stem: onnx::Conv_1440 / 1441 (weight / bias)
    copy(model.stem.weight, take("onnx::Conv_1440"), "stem.weight")
    copy(model.stem.bias, take("onnx::Conv_1441"), "stem.bias")

    # ConvNeXt blocks
    for i, blk in enumerate(model.convnext):
        pfx = f"tts.ae.decoder.convnext.{i}"
        copy(blk.dwconv.weight, take(f"{pfx}.dwconv.net.weight"), f"conv.{i}.dw.w")
        copy(blk.dwconv.bias,   take(f"{pfx}.dwconv.net.bias"),   f"conv.{i}.dw.b")
        copy(blk.norm.weight,   take(f"{pfx}.norm.norm.weight"),  f"conv.{i}.ln.w")
        copy(blk.norm.bias,     take(f"{pfx}.norm.norm.bias"),    f"conv.{i}.ln.b")
        copy(blk.pwconv1.weight, take(f"{pfx}.pwconv1.weight"),   f"conv.{i}.pw1.w")
        copy(blk.pwconv1.bias,   take(f"{pfx}.pwconv1.bias"),     f"conv.{i}.pw1.b")
        copy(blk.pwconv2.weight, take(f"{pfx}.pwconv2.weight"),   f"conv.{i}.pw2.w")
        copy(blk.pwconv2.bias,   take(f"{pfx}.pwconv2.bias"),     f"conv.{i}.pw2.b")
        copy(blk.gamma,          take(f"{pfx}.gamma"),            f"conv.{i}.gamma")

    # final BatchNorm
    copy(model.final_norm.weight,       take("tts.ae.decoder.final_norm.norm.weight"), "fn.w")
    copy(model.final_norm.bias,         take("tts.ae.decoder.final_norm.norm.bias"),   "fn.b")
    copy(model.final_norm.running_mean, take("tts.ae.decoder.final_norm.norm.running_mean"), "fn.rm")
    copy(model.final_norm.running_var,  take("tts.ae.decoder.final_norm.norm.running_var"),  "fn.rv")

    # head
    copy(model.head_layer1.weight, take("tts.ae.decoder.head.layer1.net.weight"), "h1.w")
    copy(model.head_layer1.bias,   take("tts.ae.decoder.head.layer1.net.bias"),   "h1.b")
    copy(model.head_act.weight,    take("onnx::PRelu_1505").reshape(-1),          "hp.w")
    copy(model.head_layer2.weight, take("tts.ae.decoder.head.layer2.weight"),     "h2.w")

    return warnings


# ===== Verify against onnxruntime ==================================

def verify(onnx_path: str, device: str = "cpu", seed: int = 0) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = Vocoder()
    load_vocoder_weights(model, onnx_path)
    model.eval().to(device)

    # make dummy input matching [B, 144, L]
    B, L = 2, 17
    x_np = np.random.randn(B, 144, L).astype(np.float32) * 0.3

    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    y_onnx = sess.run(None, {"latent": x_np})[0]

    with torch.no_grad():
        y_torch = model(torch.from_numpy(x_np).to(device)).cpu().numpy()

    assert y_onnx.shape == y_torch.shape, f"shape diff {y_onnx.shape} vs {y_torch.shape}"
    diff = np.abs(y_onnx - y_torch)
    return {
        "out_shape": y_onnx.shape,
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "onnx_out_range": (float(y_onnx.min()), float(y_onnx.max())),
        "torch_out_range": (float(y_torch.min()), float(y_torch.max())),
    }


if __name__ == "__main__":
    import os, json
    ONNX = os.path.join(os.path.dirname(__file__), "..", "assets", "onnx", "vocoder.onnx")
    r = verify(ONNX)
    print(json.dumps(r, indent=2))
