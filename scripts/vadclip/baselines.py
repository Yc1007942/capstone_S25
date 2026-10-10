"""Frame-level baselines on the SAME cached ViT-B/16 features, to isolate the adapter's value.

Run from the project root:

    python -m scripts.vadclip.baselines \
        --cache-dir results/vadclip/features \
        --weights data/models/openai_clip_vitb16.pt \
        --out-dir  results/vadclip/baselines

Two baselines, both scored on exactly the same valid frame-reps (build_clip at visual_length=256)
and soft labels as the VadCLIP model, so the only difference is *what* consumes the features:

    zeroshot   : frozen CLIP zero-shot -- per-frame cosine to the "normal"/"smoking" class-text
                 features, softmax over the two classes. No training, no temporal context. This is
                 the natural lower bound for a reference-free CLIP detector.
    linearprobe: logistic regression on the frozen frame features (trained on train frames only).
                 Answers "are the frozen ViT-B/16 features alone discriminative?" -- i.e. how much
                 of the model's score comes from good features vs temporal/graph modelling.

Writes baselines.json (per-baseline metrics) and scores.npz (y_true + each baseline's frame scores).
No image content is displayed; only cached feature tensors are used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from .clip_b16 import load_clip_b16, tokenize
from .data import PROMPTS, build_clip
from .metrics import frame_metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--visual-length", type=int, default=256)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    clip_model = load_clip_b16(args.weights, device).eval()

    # --- zero-shot class-text features (standard CLIP text encoder via our clean port) ----
    with torch.no_grad():
        tok = tokenize(PROMPTS)                                   # [2, 77]
        emb = clip_model.encode_token(tok)                        # [2, 77, 512] raw token embeddings
        class_feats = clip_model.encode_text(emb, tok)            # [2, 512] standard CLIP text feats
    class_feats = (class_feats / class_feats.norm(dim=-1, keepdim=True)).numpy()   # [2, 512]

    manifest = json.loads((Path(args.cache_dir) / "manifest.json").read_text(encoding="utf-8"))
    train_X, valid_rows = [], []
    for key in sorted(manifest["sessions"]):
        d = np.load(Path(args.cache_dir) / f"{key.replace(':', '_')}.npz")
        c_feat, c_label, length = build_clip(d["feat"], d["labels"], visual_length=args.visual_length)
        X = c_feat[:length]                                       # [L', 512] valid reps only
        y = c_label[:length].astype(np.float32)
        if manifest["sessions"][key]["split"] == "train":
            train_X.append((X, y))
        else:
            valid_rows.append((X, y))

    # --- zero-shot scores on valid frames -----------------------------------------------
    zs_scores = []
    for X, _ in valid_rows:
        Xt = torch.from_numpy(X) / (torch.norm(torch.from_numpy(X), dim=-1, keepdim=True) + 1e-9)
        cos = (Xt @ torch.from_numpy(class_feats.T)).numpy()      # [L', 2] cosine to normal/smoking
        e = np.exp(cos - cos.max(axis=1, keepdims=True))
        zs_scores.append((e / e.sum(axis=1, keepdims=True))[:, 1])   # P(smoking)

    y_valid = np.concatenate([y for _, y in valid_rows]).astype(int)
    zs = np.concatenate(zs_scores)

    # --- linear probe (train on train frames, score valid frames) ------------------------
    Xtr = np.concatenate([X for X, _ in train_X])
    ytr = (np.concatenate([y for _, y in train_X]) >= 0.5).astype(int)   # hard labels for the probe
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr, ytr)
    lp = clf.predict_proba(np.concatenate([X for X, _ in valid_rows]))[:, 1]   # [n_valid] P(smoking)

    metrics = {
        "zeroshot": frame_metrics(y_valid, zs),
        "linearprobe": frame_metrics(y_valid, lp),
    }
    (out_dir / "baselines.json").write_text(json.dumps(
        {"visual_length": args.visual_length, "n_train_frames": int(len(Xtr)),
         "n_valid_frames": int(len(y_valid)), "metrics": metrics}, indent=2), encoding="utf-8")
    np.savez_compressed(out_dir / "scores.npz", y_true=y_valid, score_zeroshot=zs, score_linearprobe=lp)

    print(f"[baselines] {len(Xtr)} train frames | {len(y_valid)} valid frames ({int(y_valid.sum())} pos)")
    for name in ("zeroshot", "linearprobe"):
        m = metrics[name]
        print(f"  {name:12} AUPR={m['aupr']:.4f} AUROC={m['auroc']:.4f} F1@opt={m['f1_opt']:.4f}")
    print(f"[baselines] -> {out_dir}/baselines.json, scores.npz")


if __name__ == "__main__":
    main()
