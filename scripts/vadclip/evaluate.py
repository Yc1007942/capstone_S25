"""Re-score a trained VadCLIP checkpoint on the valid split and dump frame-level scores.

Run from the project root:

    python -m scripts.vadclip.evaluate \
        --cache-dir results/vadclip/features \
        --weights data/models/openai_clip_vitb16.pt \
        --run-dir  results/vadclip/runs/full

Loads ``--run-dir/checkpoint.pt`` (state + config), rebuilds the model exactly as it was trained,
runs inference over every *valid* clip, and writes:
    eval.json   : per-variant frame metrics (visual / align / dual_mean / dual_max)
    scores.npz  : y_true, score_visual, score_align, score_dual_mean, score_dual_max  (1-D arrays
                  over all valid frames) -- used to draw ROC/PR curves in the report.

This is what makes the "actually test VadCLIP" claim concrete: real per-frame anomaly scores on a
held-out session split, scored with the canonical WSVAD frame-detection metrics. No image content
is displayed; only cached feature tensors flow through the model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .clip_b16 import load_clip_b16
from .data import PROMPTS, build_clip
from .metrics import frame_metrics
from .model import CLIPVAD


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--run-dir", required=True, help="directory containing checkpoint.pt")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    ckpt = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    device = torch.device("cpu")

    clip_model = load_clip_b16(args.weights, device).eval()
    model = CLIPVAD(num_class=2, clip_model=clip_model, device=device,
                    visual_length=cfg["visual_length"], attn_window=cfg["attn_window"],
                    visual_layers=cfg["temporal_layers"], use_graph=not cfg.get("no_graph", False))
    model.load_state_dict(ckpt["state"])
    model.eval()

    manifest = json.loads((Path(args.cache_dir) / "manifest.json").read_text(encoding="utf-8"))
    y_true, s_vis, s_al = [], [], []
    with torch.no_grad():
        for key in sorted(manifest["valid"]):
            d = np.load(Path(args.cache_dir) / f"{key.replace(':', '_')}.npz")
            c_feat, c_label, length = build_clip(d["feat"], d["labels"], visual_length=cfg["visual_length"])
            feats = torch.from_numpy(c_feat).unsqueeze(0)
            lengths = torch.tensor([length])
            _, logits1, _, logits2_raw = model(feats, None, PROMPTS, lengths)
            p_vis = torch.sigmoid(logits1.squeeze(-1))[0, :length].numpy()
            p_al = F.softmax(logits2_raw, dim=-1)[:, :, 1][0, :length].numpy()
            y_true.append(c_label[:length])
            s_vis.append(p_vis)
            s_al.append(p_al)

    y = np.concatenate(y_true).astype(int)
    vis = np.concatenate(s_vis)
    al = np.concatenate(s_al)
    dual_mean = 0.5 * (vis + al)
    dual_max = np.maximum(vis, al)

    metrics = {
        "visual": frame_metrics(y, vis),
        "align": frame_metrics(y, al),
        "dual_mean": frame_metrics(y, dual_mean),
        "dual_max": frame_metrics(y, dual_max),
    }
    (run_dir / "eval.json").write_text(json.dumps(
        {"config": cfg, "n_valid_frames": int(len(y)), "n_pos": int(y.sum()), "metrics": metrics},
        indent=2), encoding="utf-8")
    np.savez_compressed(run_dir / "scores.npz", y_true=y, score_visual=vis, score_align=al,
                        score_dual_mean=dual_mean, score_dual_max=dual_max)

    print(f"[eval] {len(y)} valid frames ({int(y.sum())} pos)")
    for name in ("visual", "align", "dual_mean", "dual_max"):
        m = metrics[name]
        print(f"  {name:10} AUPR={m['aupr']:.4f} AUROC={m['auroc']:.4f} F1@opt={m['f1_opt']:.4f} "
              f"(P={m['precision_opt']:.3f} R={m['recall_opt']:.3f}) thr={m['threshold_opt']:.3f}")
    print(f"[eval] -> {run_dir}/eval.json, scores.npz")


if __name__ == "__main__":
    main()
