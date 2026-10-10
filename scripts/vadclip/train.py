"""Train the VadCLIP adapter with frame-level binary supervision (the documented deviation).

Run from the project root:

    python -m scripts.vadclip.train \
        --cache-dir results/vadclip/features \
        --weights data/models/openai_clip_vitb16.pt \
        --out-dir  results/vadclip/runs/full

Why frame-level (not the paper's video-MIL): every clip in this corpus contains at least one
positive frame, so a clip-level weak label is all-ones and the MIL objective degenerates. The
meaningful unit is the frame/segment -- exactly what VadCLIP's headline segment-detection AP
measures -- so each representative frame is supervised with a soft target equal to the positive
fraction of its extraction block (exact 0/1 for sessions <= visual_length frames).

Both detection branches are trained:
    * visual   : BCE-with-logits on ``logits1`` (the per-frame anomaly classifier),
    * alignment: cross-entropy over the two class-text cosine similarities (``softmax(logits2_raw)``),
      which is stable because it uses the unscaled cosine in [-1, 1] rather than the /0.07 scale.

Ablation knobs: --visual-length, --attn-window, --temporal-layers, --no-graph. The best adapter
(state dict + config + valid metrics) is written to ``--out-dir``. No image content is displayed.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .clip_b16 import load_clip_b16
from .data import PROMPTS, VadClipDataset, build_clip, collate, load_smoking_sessions
from .metrics import frame_metrics
from .model import CLIPVAD


def load_clips(cache_dir: Path):
    """Load cached per-session features and assemble fixed-length clips.

    Returns (items_by_key {key: item}, manifest). Each item is {"feat","label","length"}.
    """
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    items = {}
    for key, meta in manifest["sessions"].items():
        npz = cache_dir / f"{key.replace(':', '_')}.npz"
        d = np.load(npz)
        feat, labels = d["feat"], d["labels"]
        c_feat, c_label, length = build_clip(feat, labels, visual_length=meta.get("visual_length", 256))
        items[key] = {"feat": c_feat, "label": c_label, "length": int(length), "split": meta["split"]}
    return items, manifest


def frame_scores(model, feats, lengths, device):
    """Return (p_visual [B,T], p_align [B,T]) for a batch of clips."""
    _, logits1, _, logits2_raw = model(feats.to(device), None, PROMPTS, lengths.to(device))
    p_vis = torch.sigmoid(logits1.squeeze(-1))                       # [B, T]
    p_align = F.softmax(logits2_raw, dim=-1)[:, :, 1]                # [B, T] anomaly class prob
    return p_vis.cpu(), p_align.cpu()


def evaluate_split(model, items: list[dict], device):
    """Aggregate frame metrics over a list of clips for each scoring variant."""
    all_true, all_vis, all_al = [], [], []
    model.eval()
    with torch.no_grad():
        for it in items:
            feats = torch.from_numpy(it["feat"]).unsqueeze(0)       # [1, L, W]
            lengths = torch.tensor([it["length"]])
            p_vis, p_al = frame_scores(model, feats, lengths, device)
            L = it["length"]
            all_true.append(it["label"][:L])
            all_vis.append(p_vis[0, :L].numpy())
            all_al.append(p_al[0, :L].numpy())
    y = np.concatenate(all_true)
    vis = np.concatenate(all_vis)
    al = np.concatenate(all_al)
    return {
        "visual": frame_metrics(y, vis),
        "align": frame_metrics(y, al),
        "dual_mean": frame_metrics(y, 0.5 * (vis + al)),
        "dual_max": frame_metrics(y, np.maximum(vis, al)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--visual-length", type=int, default=256)
    ap.add_argument("--attn-window", type=int, default=8)
    ap.add_argument("--temporal-layers", type=int, default=2)
    ap.add_argument("--no-graph", action="store_true", help="ablation: temporal-only (drop GCN branches)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=234)
    ap.add_argument("--patience", type=int, default=50, help="early-stop patience on valid dual AUPR")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] loading frozen ViT-B/16 from {args.weights}", flush=True)
    clip_model = load_clip_b16(args.weights, device).eval()
    model = CLIPVAD(num_class=2, clip_model=clip_model, device=device,
                    visual_length=args.visual_length, attn_window=args.attn_window,
                    visual_layers=args.temporal_layers, use_graph=not args.no_graph)

    items_by_key, manifest = load_clips(Path(args.cache_dir))
    # rebuild clips at the requested visual length (manifest may have been extracted at 256)
    for key in list(items_by_key):
        npz = Path(args.cache_dir) / f"{key.replace(':', '_')}.npz"
        d = np.load(npz)
        c_feat, c_label, length = build_clip(d["feat"], d["labels"], visual_length=args.visual_length)
        items_by_key[key] = {"feat": c_feat, "label": c_label, "length": int(length),
                             "split": manifest["sessions"][key]["split"]}

    train_items = [items_by_key[k] for k in sorted(items_by_key) if items_by_key[k]["split"] == "train"]
    valid_items = [items_by_key[k] for k in sorted(items_by_key) if items_by_key[k]["split"] == "valid"]
    n_train_frames = sum(it["length"] for it in train_items)
    n_valid_frames = sum(it["length"] for it in valid_items)
    print(f"[train] {len(train_items)} train clips ({n_train_frames} frames) | "
          f"{len(valid_items)} valid clips ({n_valid_frames} frames)", flush=True)

    loader = DataLoader(VadClipDataset(train_items), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    print(f"[train] trainable params: {n_params/1e6:.2f}M | config: L={args.visual_length} "
          f"win={args.attn_window} layers={args.temporal_layers} graph={not args.no_graph}", flush=True)

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    best_aupr, best_state, bad_epochs = float("-inf"), None, 0
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        for feats, labels, lengths in loader:
            opt.zero_grad()
            _, logits1, _, logits2_raw = model(feats, None, PROMPTS, lengths)
            L = feats.shape[1]
            mask = torch.arange(L).unsqueeze(0) < lengths.unsqueeze(1)   # [B, L] valid frames

            vis_loss = F.binary_cross_entropy_with_logits(logits1.squeeze(-1), labels, reduction="none")
            p_al = F.softmax(logits2_raw, dim=-1)[:, :, 1].clamp(1e-6, 1 - 1e-6)
            al_loss = F.binary_cross_entropy(p_al, labels, reduction="none")
            loss = (vis_loss + al_loss)[mask].mean()
            loss.backward()
            opt.step()
            run_loss += loss.item()

        val = evaluate_split(model, valid_items, device)
        dual_aupr = val["dual_mean"]["aupr"]
        history.append({"epoch": epoch, "loss": run_loss / max(1, len(loader)), **{k: v["aupr"] for k, v in val.items()}})

        if (epoch % 5 == 0 or epoch == 1):
            print(f"[train] ep {epoch:3d} loss={run_loss/max(1,len(loader)):.4f} | "
                  f"val AUPR vis={val['visual']['aupr']:.3f} al={val['align']['aupr']:.3f} "
                  f"dual={dual_aupr:.3f}", flush=True)

        if dual_aupr > best_aupr + 1e-4:
            best_aupr, bad_epochs = dual_aupr, 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save({"state": best_state, "config": vars(args), "best_epoch": epoch},
                       out_dir / "checkpoint.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"[train] early stop at epoch {epoch} (no valid-AUPR gain for {args.patience})", flush=True)
                break

    # final evaluation with the best checkpoint
    model.load_state_dict(best_state)
    final = evaluate_split(model, valid_items, device)
    result = {
        "config": vars(args),
        "n_train_clips": len(train_items), "n_valid_clips": len(valid_items),
        "n_train_frames": n_train_frames, "n_valid_frames": n_valid_frames,
        "trainable_params_M": round(n_params / 1e6, 3),
        "best_epoch": int(history[-1]["epoch"]) if history else None,
        "best_dual_aupr": best_aupr,
        "valid_metrics": final,
        "history_tail": history[-20:],
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[train] done in {time.time()-t0:.1f}s | best dual AUPR={best_aupr:.4f} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
