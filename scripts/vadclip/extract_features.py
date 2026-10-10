"""Extract and cache frozen ViT-B/16 per-frame features for the smoking corpus.

Run from the project root:

    python -m scripts.vadclip.extract_features \
        --root data/raw/datasets \
        --weights data/models/openai_clip_vitb16.pt \
        --cache-dir results/vadclip/features

For every session it loads each frame through the official CLIP transform, runs the frozen
vision tower (``encode_image`` -> raw [512] features), and writes one ``.npz`` per session:
    feat   : float32 [T, 512]   per-frame features in temporal order
    labels : int8    [T]        dense per-frame smoke label (0/1)
It also writes ``manifest.json`` with the session metadata and the deterministic
session-level train/valid split. Features are cached so training/ablations never re-run the
backbone. No image content is displayed -- only tensors flow through the model.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .clip_b16 import load_clip_b16, make_preprocess
from .data import load_smoking_sessions, make_session_split


def extract_session(model, preprocess, frames: list[str], batch_size: int = 32, device="cpu") -> np.ndarray:
    """Batched frozen-vision features for an ordered frame list -> [T, 512] float32."""
    feats = []
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            chunk = frames[i:i + batch_size]
            imgs = torch.stack([preprocess(_open(p)) for p in chunk]).to(device)
            f = model.encode_image(imgs).cpu().numpy()   # [b, 512] raw (unnormalised)
            feats.append(f.astype(np.float32))
    return np.concatenate(feats, axis=0)


def _open(path: str):
    from PIL import Image
    with Image.open(path) as im:
        return im.copy()   # decouple from the file handle before batch processing


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="datasets root (contains smoking/)")
    ap.add_argument("--weights", required=True, help="path to openai_clip_vitb16.pt")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--valid-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=234)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    print(f"[extract] loading ViT-B/16 from {args.weights}", flush=True)
    model = load_clip_b16(args.weights, device).eval()
    preprocess = make_preprocess(224)

    sessions = load_smoking_sessions(args.root)
    keys = sorted(sessions.keys())
    train_keys, valid_keys = make_session_split(keys, args.valid_frac, args.seed)
    print(f"[extract] {len(keys)} sessions | train={len(train_keys)} valid={len(valid_keys)}", flush=True)

    manifest = {"seed": args.seed, "valid_frac": args.valid_frac,
                "train": train_keys, "valid": valid_keys, "sessions": {}}
    t0 = time.time()
    for n, key in enumerate(keys):
        s = sessions[key]
        out = cache_dir / f"{key.replace(':', '_')}.npz"
        if out.is_file():
            data = np.load(out)
            feat, labels = data["feat"], data["labels"]
        else:
            feat = extract_session(model, preprocess, s.frames, args.batch_size, device)
            labels = np.asarray(s.labels, dtype=np.int8)
            assert feat.shape[0] == len(labels), f"{key}: {feat.shape} vs {len(labels)}"
            np.savez_compressed(out, feat=feat, labels=labels)
        manifest["sessions"][key] = {
            "n_frames": int(feat.shape[0]),
            "n_positive": int(labels.sum()),
            "split": "train" if key in train_keys else "valid",
        }
        print(f"[extract] ({n+1}/{len(keys)}) {key:<24} T={feat.shape[0]:<5} pos={int(labels.sum()):<4}", flush=True)

    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[extract] done in {time.time()-t0:.1f}s -> {cache_dir}", flush=True)


if __name__ == "__main__":
    main()
