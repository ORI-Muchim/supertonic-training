"""KSS (Korean Single-speaker Speech) dataset loader.

Source: https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset
Expected layout (already extracted into `archive/` in this repo):
    archive/
    ├── transcript.v.1.4.txt          # | delimited: path | raw | norm1 | norm2 | duration | english
    └── kss/
        ├── 1/ 1_0000.wav ...         # 44.1 kHz stereo (essentially mono), float64
        ├── 2/ ...
        ├── 3/ ...
        └── 4/ ...

Usage:
    # 1. One-time: build index file (scans all wavs, saves metadata)
    python -m training.data.kss --build_index
    # 2. In training code:
    ds = KSSDataset(index_path="training/data/kss_index.json", crop_seconds=1.0)
    dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True, num_workers=2)
    for wav in dl:   # wav: [B, 44100]
        ...

Stages that use this:
    - AE:   only `wav` (random crops)
    - TTL:  full utterance + text + language tag
    - DP:   full utterance duration + text
"""
from __future__ import annotations
import os, json, random, argparse
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


SAMPLE_RATE = 44100
DEFAULT_ARCHIVE = Path(__file__).resolve().parents[2] / "archive"
DEFAULT_INDEX   = Path(__file__).resolve().parent / "kss_index.json"


# ------------------------- index builder ------------------------- #
@dataclass
class KSSEntry:
    path: str         # relative to archive/kss/
    duration: float   # seconds
    text_raw: str     # raw Korean (with punctuation)
    text_norm: str    # normalized (numbers expanded etc.)


def build_index(
    archive_dir: Path = DEFAULT_ARCHIVE,
    transcript: str = "transcript.v.1.4.txt",
    out_path: Path = DEFAULT_INDEX,
    verify_wavs: bool = True,
    min_seconds: float = 0.5,
    max_seconds: float = 12.0,
):
    """Scan transcript and (optionally) verify wav files exist and have correct sample rate.
    Writes JSON: {"entries": [...], "sample_rate": 44100, ...}
    """
    archive_dir = Path(archive_dir)
    transcript_path = archive_dir / transcript
    kss_root = archive_dir / "kss"
    assert transcript_path.exists(), f"not found: {transcript_path}"
    assert kss_root.exists(), f"not found: {kss_root}"

    entries: list[dict] = []
    skipped = {"missing": 0, "sr_mismatch": 0, "too_short": 0, "too_long": 0, "parse": 0}
    total = 0

    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 5:
                skipped["parse"] += 1
                continue
            rel_path, raw, norm1, norm2, dur_str = parts[:5]
            try:
                duration = float(dur_str)
            except ValueError:
                skipped["parse"] += 1
                continue

            wav_path = kss_root / rel_path
            if verify_wavs and not wav_path.exists():
                skipped["missing"] += 1
                continue

            if verify_wavs:
                info = sf.info(str(wav_path))
                if info.samplerate != SAMPLE_RATE:
                    skipped["sr_mismatch"] += 1
                    continue
                # use actual wav duration if transcript differs by more than 0.1 s
                actual_dur = info.frames / info.samplerate
                if abs(actual_dur - duration) > 0.1:
                    duration = actual_dur

            if duration < min_seconds:
                skipped["too_short"] += 1
                continue
            if duration > max_seconds:
                skipped["too_long"] += 1
                continue

            entries.append(asdict(KSSEntry(
                path=rel_path, duration=duration,
                text_raw=raw, text_norm=norm2 or norm1 or raw,
            )))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    index = {
        "sample_rate": SAMPLE_RATE,
        "archive_dir": str(archive_dir),
        "kss_rel_root": "kss",
        "total_transcript_lines": total,
        "kept": len(entries),
        "skipped": skipped,
        "min_seconds": min_seconds,
        "max_seconds": max_seconds,
        "entries": entries,
    }
    total_dur = sum(e["duration"] for e in entries)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)

    print(f"KSS index: {len(entries)} / {total} utterances kept")
    print(f"  skipped: {skipped}")
    print(f"  total audio: {total_dur/3600:.2f} h")
    print(f"  saved to: {out_path}")
    return index


# ------------------------- Dataset ------------------------- #
class KSSDataset(Dataset):
    """Random-crop dataset for AE training.

    Returns a tensor `wav [crop_samples]` per __getitem__.
    Mono: computed as mean of 2 channels (KSS is near-identical L/R).
    """
    def __init__(
        self,
        index_path: Path | str = DEFAULT_INDEX,
        crop_seconds: float = 1.0,
        sample_rate: int = SAMPLE_RATE,
        return_text: bool = False,
        seed: int | None = None,
    ):
        index_path = Path(index_path)
        with open(index_path, "r", encoding="utf-8") as f:
            idx = json.load(f)
        assert idx["sample_rate"] == sample_rate, f"sr mismatch: {idx['sample_rate']} vs {sample_rate}"
        self.sample_rate = sample_rate
        self.archive_dir = Path(idx["archive_dir"])
        self.kss_root = self.archive_dir / idx["kss_rel_root"]
        self.entries = idx["entries"]
        self.crop_samples = int(round(crop_seconds * sample_rate))
        self.return_text = return_text
        self._seed = seed   # None → use torch generator via default random_state below

    def __len__(self):
        return len(self.entries)

    def _load_wav(self, rel_path: str) -> np.ndarray:
        wav, _sr = sf.read(str(self.kss_root / rel_path), dtype="float32")
        if wav.ndim == 2:
            wav = wav.mean(axis=1)  # stereo -> mono
        return wav

    def __getitem__(self, i: int):
        entry = self.entries[i]
        wav = self._load_wav(entry["path"])
        # Use numpy's default RNG (worker-safe via torch.utils.data.get_worker_info).
        if len(wav) >= self.crop_samples:
            max_start = len(wav) - self.crop_samples
            start = int(np.random.randint(0, max_start + 1)) if max_start > 0 else 0
            wav = wav[start:start + self.crop_samples]
        else:
            # pad with zeros (should be rare given min_seconds filter)
            pad = self.crop_samples - len(wav)
            wav = np.pad(wav, (0, pad), mode="constant")
        wav_t = torch.from_numpy(wav)
        if self.return_text:
            return wav_t, entry["text_raw"]
        return wav_t


class KSSFullUtteranceDataset(Dataset):
    """Full-utterance dataset (for TTL / DP / latent caching). Returns variable-length tensors
    — use a custom collate_fn with padding if you need batching here."""
    def __init__(self, index_path: Path | str = DEFAULT_INDEX, sample_rate: int = SAMPLE_RATE):
        with open(Path(index_path), "r", encoding="utf-8") as f:
            idx = json.load(f)
        self.sample_rate = sample_rate
        self.archive_dir = Path(idx["archive_dir"])
        self.kss_root = self.archive_dir / idx["kss_rel_root"]
        self.entries = idx["entries"]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        entry = self.entries[i]
        wav, _sr = sf.read(str(self.kss_root / entry["path"]), dtype="float32")
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        return {
            "wav": torch.from_numpy(wav),
            "duration": float(entry["duration"]),
            "text_raw": entry["text_raw"],
            "text_norm": entry["text_norm"],
            "path": entry["path"],
        }


# ------------------------- CLI ------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build_index", action="store_true")
    ap.add_argument("--archive_dir", type=str, default=str(DEFAULT_ARCHIVE))
    ap.add_argument("--out_path", type=str, default=str(DEFAULT_INDEX))
    ap.add_argument("--min_seconds", type=float, default=0.5)
    ap.add_argument("--max_seconds", type=float, default=12.0)
    ap.add_argument("--test_load", action="store_true")
    args = ap.parse_args()

    if args.build_index:
        build_index(
            archive_dir=Path(args.archive_dir),
            out_path=Path(args.out_path),
            min_seconds=args.min_seconds,
            max_seconds=args.max_seconds,
        )
    if args.test_load:
        ds = KSSDataset(args.out_path, crop_seconds=1.0, seed=0)
        print(f"dataset size: {len(ds)}")
        w = ds[0]
        print(f"sample[0] shape: {tuple(w.shape)}, dtype: {w.dtype}")
        print(f"  range: [{w.min():.3f}, {w.max():.3f}], abs_mean: {w.abs().mean():.4f}")
        # quick DataLoader test
        from torch.utils.data import DataLoader
        dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
        for b in dl:
            print(f"batch shape: {tuple(b.shape)}")
            break
