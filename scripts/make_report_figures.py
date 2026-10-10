"""Generate the report's figures from benchmark METRICS ONLY.

Every chart here plots a number that came out of the benchmark (accuracy, F1, AUC, latency,
throughput, peak RSS) or of the VadCLIP ablation suite (AUPR / AUROC / F1@opt). No dataset image is
opened, decoded, or displayed anywhere in this script -- it reads only ``improvements.json``,
``comparison.csv`` and the VadCLIP ``eval.json``/``baselines.json`` files. This keeps the figures
consistent with the "no screenshots / no direct image inspection" constraint: a reader can verify
every bar against the CSV/JSON on disk, and none of them could have been produced by looking at an
image.

Run from the project root:

    python -m scripts.make_report_figures \
        --results-dir results/anomaly_headtohead \
        --vadclip-dir  results/vadclip \
        --out-dir      reports/figures

Writes one PNG per figure (fig01..fig07). Uses the non-interactive Agg backend.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: never opens a display, never touches an image file
import matplotlib.pyplot as plt
import numpy as np

# A small, colour-blind-safe-ish palette (one hue per model / variant).
MODEL_COLORS = {
    "mobileclip2-s2": "#4C72B0",
    "siglip2-b16-224": "#DD8452",
    "smolvlm-500m": "#937860",
    "minicpm-v4": "#55A868",
}
VARIANT_COLORS = {
    "zeroshot": "#BBBBBB",
    "linearprobe": "#8C8C8C",
    "full": "#4C72B0",
    "no_graph": "#DD8452",
    "win_1": "#55A868",
    "win_16": "#C44E52",
    "vl_128": "#8172B3",
    "vl_512": "#CCB974",
}


def _load_improvements(results_dir: Path) -> dict:
    return json.loads((results_dir / "improvements.json").read_text(encoding="utf-8"))


def _load_comparison(results_dir: Path) -> list[dict]:
    with (results_dir / "comparison.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:  # numericise the columns we plot
        for k in ("latency_mean_seconds", "active_samples_per_second", "peak_rss_mb"):
            r[k] = float(r[k])
    return rows


def _ci_bar(ax, xs, means, cis, colors, ylabel, title, ylim=None):
    """Grouped bars with 95% CI error bars; skips (draws a hatch) where the value is undefined."""
    for x, m, c, col in zip(xs, means, cis, colors):
        if m is None:
            ax.bar(x, 0.02, width=0.6, color="none", edgecolor="#999", hatch="//", linewidth=1)
            ax.text(x, 0.03, "n/a", ha="center", va="bottom", fontsize=8, color="#555")
            continue
        if c is None:
            ax.bar(x, m, width=0.6, color=col)
        else:
            lo, hi = c[0], c[1]
            ax.errorbar([x], [m], yerr=[[m - lo], [hi - m]], fmt="none", ecolor="#333",
                        elinewidth=1.4, capsize=5)
            ax.bar(x, m, width=0.6, color=col, alpha=0.92)
        ax.text(x, (hi if c else m) + 0.01, f"{m:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)


def fig_accuracy_macrof1(imp: dict, out: Path):
    models = list(imp["models"])
    xs = np.arange(len(models))
    acc = [imp["models"][m]["point"]["accuracy_including_failures"] for m in models]
    acc_ci = [imp["models"][m]["ci95"]["accuracy_including_failures"] for m in models]
    mf1 = [imp["models"][m]["point"]["macro_f1"] for m in models]
    mf1_ci = [imp["models"][m]["ci95"]["macro_f1"] for m in models]

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    _ci_bar(ax, xs - 0.2, acc, acc_ci, ["#4C72B0"] * len(models), "accuracy (incl. failures)",
            "Accuracy including failures — point estimate + bootstrap 95% CI")
    ax.set_xticks(xs)
    ax.set_xticklabels([m.replace("-", "\n", 1) for m in models], fontsize=8)
    # overlay macro-F1 as a second series of markers so both headline metrics share one axis
    ax2 = ax.twinx()
    ax2.plot(xs + 0.2, mf1, marker="D", color="#C44E52", lw=0, ms=7, label="macro-F1")
    for x, m in zip(xs + 0.2, mf1):
        if m is not None:
            ax2.annotate(f"{m:.3f}", (x, m), textcoords="offset points", xytext=(0, 8),
                         ha="center", fontsize=7, color="#C44E52")
    ax2.set_ylabel("macro-F1 (diamonds)", color="#C44E52")
    ax2.tick_params(axis="y", labelcolor="#C44E52")
    fig.tight_layout()
    fig.savefig(out / "fig01_accuracy_macrof1_ci.png", dpi=150)
    plt.close(fig)


def fig_binary_f1_auc(imp: dict, out: Path):
    models = list(imp["models"])
    xs = np.arange(len(models))
    bf1 = [imp["models"][m]["point"]["binary_f1"] for m in models]
    bf1_ci = [imp["models"][m]["ci95"]["binary_f1"] for m in models]
    auc = [imp["models"][m]["point"]["roc_auc"] for m in models]
    auc_ci = [imp["models"][m]["ci95"]["roc_auc"] for m in models]

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(8.6, 4.0))
    _ci_bar(axa, xs, bf1, bf1_ci, [MODEL_COLORS[m] for m in models], "binary anomaly F1",
            "Binary anomaly-vs-normal F1 (95% CI)")
    axa.set_xticks(xs); axa.set_xticklabels([m.replace("-", "\n", 1) for m in models], fontsize=7)
    _ci_bar(axb, xs, auc, auc_ci, [MODEL_COLORS[m] for m in models], "ROC-AUC (anomaly score)",
            "Binary ROC-AUC — threshold-free scoring quality (95% CI)")
    axb.set_xticks(xs); axb.set_xticklabels([m.replace("-", "\n", 1) for m in models], fontsize=7)
    fig.suptitle("SmolVLM abstains on all 80 samples → no binary score; MiniCPM-V emits no numeric "
                 "anomaly score (AUC undefined)", fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "fig02_binary_f1_auc_ci.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_perclass_heatmap(imp: dict, out: Path):
    labels = imp["labels"]
    models = list(imp["models"])
    M = np.full((len(models), len(labels)), np.nan)
    for i, m in enumerate(models):
        pc = imp["models"][m].get("per_class", {})
        for j, lab in enumerate(labels):
            if lab in pc:
                M[i, j] = pc[lab]["f1"]

    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(range(len(models))); ax.set_yticklabels([m.replace("-", "\n", 1) for m in models], fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if 0.3 < (v or 0) < 0.9 else "black")
    fig.colorbar(im, ax=ax, label="per-class F1")
    ax.set_title("Per-class F1 (support = 16 per class; unattended_items has support 0 in the head-to-head set)")
    fig.tight_layout()
    fig.savefig(out / "fig03_perclass_f1_heatmap.png", dpi=150)
    plt.close(fig)


def fig_latency_throughput(comp: list[dict], out: Path):
    rows = sorted(comp, key=lambda r: -r["active_samples_per_second"])
    names = [r["model"].replace("-", "\n", 1) for r in rows]
    lat = [r["latency_mean_seconds"] for r in rows]
    thr = [r["active_samples_per_second"] for r in rows]

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(8.6, 4.0))
    y = np.arange(len(rows))[::-1]
    bars = axa.barh(y, lat, color=[MODEL_COLORS[r["model"]] for r in rows])
    axa.set_yticks(y); axa.set_yticklabels(names, fontsize=8)
    axa.set_xscale("log"); axa.set_xlabel("mean latency per sample (s, log scale)")
    axa.set_title("Inference latency")
    for yi, v in zip(y, lat):
        axa.text(v * 1.05, yi, f"{v:.3f}s", va="center", fontsize=8)

    bars = axb.barh(y, thr, color=[MODEL_COLORS[r["model"]] for r in rows])
    axb.set_yticks(y); axb.set_yticklabels(names, fontsize=8)
    axb.set_xscale("log"); axb.set_xlabel("active samples / second (log scale)")
    axb.set_title("Throughput")
    for yi, v in zip(y, thr):
        axb.text(v * 1.05, yi, f"{v:.2f}/s", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig04_latency_throughput.png", dpi=150)
    plt.close(fig)


def fig_peak_rss(comp: list[dict], out: Path):
    rows = sorted(comp, key=lambda r: -r["peak_rss_mb"])
    names = [r["model"].replace("-", "\n", 1) for r in rows]
    rss = [r["peak_rss_mb"] / 1024.0 for r in rows]  # MB -> GB

    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    bars = ax.bar(names, rss, color=[MODEL_COLORS[r["model"]] for r in rows])
    ax.set_ylabel("peak RSS (GiB)")
    ax.set_title("Peak resident memory per model (load + inference)")
    for b, v, r in zip(bars, rss, rows):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{r['peak_rss_mb']/1024:.2f} GiB",
                ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig05_peak_rss.png", dpi=150)
    plt.close(fig)


def _vadclip_metrics(vad_dir: Path) -> dict[str, dict]:
    """Return {variant: {branch: metrics}} for baselines + all runs."""
    out = {}
    bl = json.loads((vad_dir / "baselines" / "baselines.json").read_text(encoding="utf-8"))
    for name in ("zeroshot", "linearprobe"):
        out[name] = {"score": bl["metrics"][name]}  # single score, not a dual branch
    for run in ["full", "no_graph", "win_1", "win_16", "vl_128", "vl_512"]:
        d = json.loads((vad_dir / "runs" / run / "eval.json").read_text(encoding="utf-8"))
        out[run] = {b: m for b, m in d["metrics"].items()}
    return out


def fig_vadclip_ablation(vad_dir: Path, out: Path):
    M = _vadclip_metrics(vad_dir)
    variants = ["zeroshot", "linearprobe", "full", "no_graph", "win_1", "win_16", "vl_128", "vl_512"]

    def best_aupr(v):
        if v in ("zeroshot", "linearprobe"):
            return M[v]["score"]["aupr"]
        # dual_mean is the model's intended combined score; report that as the headline AUPR
        return M[v]["dual_mean"]["aupr"]

    def best_auroc(v):
        if v in ("zeroshot", "linearprobe"):
            return M[v]["score"]["auroc"]
        return M[v]["dual_mean"]["auroc"]

    xs = np.arange(len(variants))
    aupr = [best_aupr(v) for v in variants]
    auroc = [best_auroc(v) for v in variants]
    colors = [VARIANT_COLORS[v] for v in variants]

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    w = 0.38
    ax.bar(xs - w / 2, aupr, width=w, color=colors, alpha=0.95, label="AUPR (dual_mean)")
    ax.bar(xs + w / 2, auroc, width=w, color=colors, alpha=0.45, hatch="//", label="AUROC (dual_mean)")
    for x, a in zip(xs - w / 2, aupr):
        ax.text(x, a + 0.01, f"{a:.3f}", ha="center", fontsize=7)
    for x, a in zip(xs + w / 2, auroc):
        ax.text(x, a + 0.01, f"{a:.3f}", ha="center", fontsize=7)
    ax.axhline(0.5, color="#C44E52", ls="--", lw=1, alpha=0.6)
    ax.text(len(variants) - 0.4, 0.505, "chance (AUROC)", fontsize=7, color="#C44E52")
    ax.set_xticks(xs); ax.set_xticklabels(variants, fontsize=8)
    ax.set_ylabel("frame-level AUPR / AUROC"); ax.set_ylim(0, 0.9)
    ax.legend(fontsize=8)
    ax.set_title("VadCLIP ablation — dual_mean branch (baselines are single-score; see §6 for per-branch detail)")
    fig.tight_layout()
    fig.savefig(out / "fig06_vadclip_ablation_aupr.png", dpi=150)
    plt.close(fig)


def fig_vadclip_branch_breakdown(vad_dir: Path, out: Path):
    M = _vadclip_metrics(vad_dir)
    branches = ["visual", "align", "dual_mean", "dual_max"]
    variants = ["full", "no_graph", "win_1", "win_16", "vl_128", "vl_512"]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9), sharey=True)
    for ax, metric in zip(axes, ("aupr", "auroc")):
        xs = np.arange(len(variants))
        w = 0.8 / len(branches)
        branch_colors = ["#4C72B0", "#55A868", "#DD8452", "#937860"]
        for bi, br in enumerate(branches):
            vals = [M[v][br][metric] for v in variants]
            ax.bar(xs + (bi - 1.5) * w, vals, width=w, color=branch_colors[bi], label=br)
        ax.set_xticks(xs); ax.set_xticklabels(variants, fontsize=8)
        ax.axhline(0.5, color="#C44E52", ls="--", lw=1, alpha=0.6)
        ax.set_ylabel("frame-level " + ("AUPR" if metric == "aupr" else "AUROC"))
        ax.set_title(f"{metric.upper()} by detection branch")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("VadCLIP dual-branch breakdown (vl_128/vl_512 use a different valid frame set — see §6.7)",
                 fontsize=9, y=1.03)
    fig.tight_layout()
    fig.savefig(out / "fig07_vadclip_branch_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--vadclip-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default="reports/figures")
    args = ap.parse_args(argv)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    imp = _load_improvements(args.results_dir)
    comp = _load_comparison(args.results_dir)

    fig_accuracy_macrof1(imp, out)
    fig_binary_f1_auc(imp, out)
    fig_perclass_heatmap(imp, out)
    fig_latency_throughput(comp, out)
    fig_peak_rss(comp, out)
    fig_vadclip_ablation(args.vadclip_dir, out)
    fig_vadclip_branch_breakdown(args.vadclip_dir, out)

    print(f"[figures] wrote 7 PNGs -> {out}")
    for p in sorted(out.glob("*.png")):
        print(f"   {p.name}  ({p.stat().st_size/1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
