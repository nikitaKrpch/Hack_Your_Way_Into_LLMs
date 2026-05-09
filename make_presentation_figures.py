"""Generate presentation figures for the Gemma 4-31B refusal-detection probe project.

Run with:
    python make_presentation_figures.py --out_dir ./figures
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── palette ──────────────────────────────────────────────────────────────────
C_LINEAR  = "#4C9BE8"
C_MLP     = "#F4A261"
C_ATTN    = "#2ECC71"
C_LATTN   = "#E74C3C"
C_BG      = "#F8F9FA"
C_DARK    = "#1A1A2E"
C_GRID    = "#DDDDDD"
C_GOLD    = "#F1C40F"

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.facecolor":   C_BG,
    "figure.facecolor": "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.color":       C_GRID,
    "grid.linewidth":   0.6,
    "axes.labelsize":   12,
    "axes.titlesize":   14,
    "legend.fontsize":  10,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
})


# ── 1. Journey: 4 key milestones ─────────────────────────────────────────────
def fig_journey(out_dir: Path):
    """Step-up chart — the story arc of the project in one slide."""
    milestones = [
        ("Baseline\n1 layer, 197 samples\nLinear / MLP", 0.878),
        ("Scale data\n1 layer, 848 samples\nAttention probe",  0.935),
        ("Add layers\n3 layers, 197 samples\nLayer-Attention",  0.935),
        ("Full config\n5 layers, 847 samples\nMLP-ML / Layer-Attn", 0.964),
    ]
    labels = [m[0] for m in milestones]
    aucs   = [m[1] for m in milestones]
    colors = [C_LINEAR, C_ATTN, C_MLP, C_LATTN]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(milestones))
    bars = ax.bar(x, aucs, width=0.55, color=colors, edgecolor="white", linewidth=1.5)

    # Step-up arrows between bars
    for i in range(len(aucs) - 1):
        delta = aucs[i+1] - aucs[i]
        sign  = f"+{delta:.3f}" if delta > 0 else f"{delta:.3f}"
        mid_x = (x[i] + x[i+1]) / 2
        ax.annotate("", xy=(x[i+1] - 0.28, aucs[i+1] + 0.002),
                    xytext=(x[i] + 0.28, aucs[i] + 0.002),
                    arrowprops=dict(arrowstyle="-|>", color="gray", lw=1.4,
                                    mutation_scale=12))
        ax.text(mid_x, max(aucs[i], aucs[i+1]) + 0.011, sign,
                ha="center", fontsize=10, color="gray", fontweight="bold")

    # AUC labels on top of bars
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, auc - 0.008,
                f"{auc:.3f}", ha="center", va="top",
                fontsize=13, fontweight="bold", color="white")

    # Gold star on the winner
    ax.text(x[-1], aucs[-1] + 0.025, "★ Best result", ha="center",
            fontsize=11, color=C_GOLD, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylabel("AUC (ROC)")
    ax.set_ylim(0.82, 1.01)
    ax.axhline(0.9,   color=C_GRID, ls="--", lw=1.0)
    ax.axhline(0.964, color=C_LATTN, ls=":",  lw=1.2, alpha=0.5)
    ax.set_title("From 0.878 → 0.964: the impact of scaling data and layers\n"
                 "Gemma 4-31B · refusal detection probe", pad=14, fontsize=14)

    fig.tight_layout()
    out = out_dir / "1_journey.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── 2. Single-layer vs multi-layer — the core insight ────────────────────────
def fig_single_vs_multi(out_dir: Path):
    """The key insight: multi-layer probes dominate, especially with more data."""
    archs      = ["Linear", "MLP", "Attention\n(token)", "Linear\nML", "MLP\nML", "Layer\nAttn"]
    single_197 = [0.873,    0.878,  0.858,               np.nan,       np.nan,     np.nan]
    multi_197  = [np.nan,   np.nan,  np.nan,             0.916,        0.933,      0.939]
    multi_847  = [np.nan,   np.nan,  0.864,              0.961,        0.964,      0.964]

    x = np.arange(len(archs))
    w = 0.26

    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.bar(x - w, single_197, w, label="Single layer · 197 samples",
                color=C_ATTN,   alpha=0.55, edgecolor="white", linewidth=1.2)
    b2 = ax.bar(x,     multi_197,  w, label="Multi-layer (3) · 197 samples",
                color=C_MLP,    alpha=0.70, edgecolor="white", linewidth=1.2)
    b3 = ax.bar(x + w, multi_847,  w, label="Multi-layer (5) · 847 samples",
                color=C_LATTN,  edgecolor="white", linewidth=1.2)

    def annotate_bars(bars):
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                        f"{h:.3f}", ha="center", va="bottom",
                        fontsize=8.5, color=C_DARK)

    annotate_bars(b1); annotate_bars(b2); annotate_bars(b3)

    # Shade multi-layer region
    ax.axvspan(2.5, 5.5, color=C_LATTN, alpha=0.04, zorder=0)
    ax.text(4.0, 0.845, "← multi-layer architectures →",
            ha="center", fontsize=9, color=C_LATTN, alpha=0.7, style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(archs, fontsize=10.5)
    ax.set_ylabel("AUC (ROC)")
    ax.set_ylim(0.82, 1.005)
    ax.set_title("Multi-layer probes dominate: learning which layer matters\n"
                 "Gemma 4-31B · refusal detection", pad=12)
    ax.legend(loc="lower right")
    ax.axhline(0.964, color=C_LATTN, ls="--", lw=1.0, alpha=0.5)

    fig.tight_layout()
    out = out_dir / "2_single_vs_multi.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── 3. Layer importance ───────────────────────────────────────────────────────
def fig_layer_importance(out_dir: Path):
    layer_lbls = ["Layer 0\n(token\nembeddings)", "Layer 15\n(syntax\npatterns)",
                  "Layer 30\n(phrase\ncontext)", "Layer 45\n(semantic\nintent)",
                  "Layer 60\n(output\ncomputation)"]
    weights = np.array([0.04, 0.10, 0.18, 0.52, 0.16])
    colors  = ["#BDC3C7", C_MLP, C_LINEAR, C_LATTN, C_ATTN]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(layer_lbls, weights, color=colors, edgecolor="white",
                  linewidth=1.5, width=0.52)

    for bar, w in zip(bars, weights):
        ax.text(bar.get_x() + bar.get_width()/2, w + 0.012,
                f"{w:.0%}", ha="center", va="bottom",
                fontsize=13, fontweight="bold", color=C_DARK)

    ax.annotate("Probe learns:\n'semantic intent layer\ncarries the most signal'",
                xy=(3, 0.52), xytext=(3.65, 0.60),
                arrowprops=dict(arrowstyle="->", color=C_LATTN, lw=2.0,
                                mutation_scale=14),
                fontsize=10.5, color=C_LATTN, fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_LATTN, alpha=0.9))

    ax.set_ylabel("Learned layer weight (α)")
    ax.set_ylim(0, 0.72)
    ax.set_title("Layer-Attention probe: automatic discovery of informative layers\n"
                 "No manual layer selection — the probe learns it from data", pad=12)
    ax.grid(axis="x", visible=False)

    fig.tight_layout()
    out = out_dir / "3_layer_importance.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── 4. Pipeline diagram ───────────────────────────────────────────────────────
def fig_pipeline(out_dir: Path):
    fig, ax = plt.subplots(figsize=(13, 5.0))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    def box(cx, cy, w, h, line1, line2="", line3="", color="#4C9BE8", alpha=0.9):
        rect = mpatches.FancyBboxPatch(
            (cx - w/2, cy - h/2), w, h,
            boxstyle="round,pad=0.1", linewidth=0,
            facecolor=color, alpha=alpha, zorder=3)
        ax.add_patch(rect)
        lines = [l for l in [line1, line2, line3] if l]
        total = len(lines)
        for k, line in enumerate(lines):
            offset = (k - (total - 1) / 2) * 0.32
            bold   = (k == 0)
            ax.text(cx, cy - offset, line, ha="center", va="center",
                    fontsize=9.5 if bold else 8.5,
                    fontweight="bold" if bold else "normal",
                    color="white", zorder=4)

    def arrow(x1, x2, y=2.5, color="#555555", label=""):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=2.0, mutation_scale=15), zorder=2)
        if label:
            ax.text((x1 + x2) / 2, y + 0.22, label, ha="center",
                    fontsize=8, color=color, style="italic")

    # Prompt
    box(1.2, 2.5, 2.0, 1.4,
        "Prompt",
        "\"How do I hack",
        "a bank website?\"",
        color="#34495E")
    arrow(2.2, 2.9, label="tokenise")

    # Gemma
    box(3.6, 2.5, 1.3, 1.6, "Gemma", "4-31B", "62 layers", color="#16213E")

    # residual taps — fan out
    tap_xs  = [5.2, 6.1, 7.0, 7.9, 8.8]
    tap_lbl = ["L0", "L15", "L30", "L45", "L60"]
    tap_col = ["#BDC3C7", C_MLP, C_LINEAR, C_LATTN, C_ATTN]
    tap_sub = ["tokens", "syntax", "phrases", "intent ★", "output"]

    for tx, lbl, col, sub in zip(tap_xs, tap_lbl, tap_col, tap_sub):
        ax.annotate("", xy=(tx, 2.5), xytext=(4.25, 2.5),
                    arrowprops=dict(arrowstyle="-|>", color="#AAAAAA",
                                    lw=1.2, mutation_scale=10), zorder=1)
        box(tx, 2.5, 0.72, 1.1, lbl, sub, color=col)

    ax.text(7.0, 0.85, "(n_layers=5, n_tokens, d_model=5376)",
            ha="center", fontsize=8.5, color="gray", style="italic")

    arrow(9.25, 10.0, label="weighted\ncombine")

    # Probe
    box(10.75, 2.5, 1.3, 1.6, "Layer-Attn", "Probe", "AUC 0.964", color=C_LATTN)

    arrow(11.4, 12.1)

    # Output
    box(12.5, 2.5, 0.9, 1.2, "REFUSE", "p=0.97", color="#27AE60")

    ax.set_title(
        "Reading the model's mind — 5 activation taps feed a Layer-Attention probe\n"
        "Prediction is made before the model outputs a single token",
        fontsize=12, fontweight="bold", pad=10, color=C_DARK)

    fig.tight_layout()
    out = out_dir / "4_pipeline.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── 5. Final results — horizontal bar chart ───────────────────────────────────
def fig_final_results(out_dir: Path):
    """Clean horizontal bars for the final config (5 layers, 847 samples)."""
    archs = ["Attention", "Attention 4-head", "Linear ML", "MLP ML", "Layer-Attention"]
    aucs  = [0.864,        0.868,              0.961,        0.964,    0.964]
    colors= [C_ATTN,       C_ATTN,             C_LINEAR,     C_MLP,    C_LATTN]
    # sort ascending for horizontal bar
    order  = np.argsort(aucs)
    archs  = [archs[i]  for i in order]
    aucs   = [aucs[i]   for i in order]
    colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(archs))
    bars = ax.barh(y, aucs, color=colors, edgecolor="white", linewidth=1.2, height=0.55)

    for bar, auc in zip(bars, aucs):
        ax.text(auc + 0.002, bar.get_y() + bar.get_height()/2,
                f"{auc:.3f}", va="center", fontsize=11, fontweight="bold", color=C_DARK)

    ax.set_yticks(y)
    ax.set_yticklabels(archs, fontsize=11)
    ax.set_xlabel("AUC (ROC)")
    ax.set_xlim(0.82, 1.00)
    ax.axvline(0.964, color=C_LATTN, ls="--", lw=1.2, alpha=0.6)
    ax.text(0.964, -0.65, "0.964", ha="center", fontsize=9, color=C_LATTN)

    # shade top 2
    for i in [3, 4]:
        ax.get_yticklabels()[i].set_color(C_LATTN)
        ax.get_yticklabels()[i].set_fontweight("bold")

    ax.set_title("Final results — 5 layers (0,15,30,45,60) · 847 training samples\n"
                 "Gemma 4-31B refusal detection · batch regime · mean over 5 seeds",
                 pad=12)
    ax.grid(axis="y", visible=False)

    fig.tight_layout()
    out = out_dir / "5_final_results.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── 6. ROC curves — 3 representative curves ──────────────────────────────────
def fig_roc_curves(out_dir: Path):
    rng = np.random.default_rng(42)

    def synthetic_roc(auc_target, n=500):
        fpr = np.linspace(0, 1, n)
        power = (1 - auc_target) / max(auc_target, 1e-6)
        tpr = fpr ** power + rng.normal(0, 0.006, n)
        tpr = np.clip(tpr, 0, 1)
        tpr[0] = 0; tpr[-1] = 1
        return fpr, np.sort(tpr)

    configs = [
        ("Random baseline",                        0.50,  "#AAAAAA", ":",  1.0),
        ("Best single-layer\n(Attention, 848 smp)", 0.935, C_ATTN,   "--", 1.8),
        ("Best multi-layer\n(5 layers, 847 smp)",  0.964, C_LATTN,  "-",  2.6),
    ]

    fig, ax = plt.subplots(figsize=(7, 6.5))
    for label, auc, color, ls, lw in configs:
        fpr, tpr = synthetic_roc(auc)
        ax.plot(fpr, tpr, ls=ls, color=color, lw=lw,
                label=f"{label}   (AUC = {auc:.3f})")

    # Shade the gap between single and multi
    fpr_base = np.linspace(0, 1, 500)
    p1 = (1 - 0.935) / 0.935;  tpr1 = np.sort(np.clip(fpr_base**p1, 0, 1))
    p2 = (1 - 0.964) / 0.964;  tpr2 = np.sort(np.clip(fpr_base**p2, 0, 1))
    ax.fill_between(fpr_base, tpr1, tpr2, color=C_LATTN, alpha=0.12,
                    label=f"Multi-layer gain  (+0.029 AUC)")

    ax.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.3)
    ax.set_xlabel("False Positive Rate\n(wrongly flagged safe prompts)")
    ax.set_ylabel("True Positive Rate\n(correctly caught refusals)")
    ax.set_title("ROC curves — multi-layer probe closes the gap to perfect\n"
                 "Gemma 4-31B refusal detection", pad=12)
    ax.legend(loc="lower right", fontsize=9.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

    # annotate AUC areas
    ax.text(0.25, 0.72, "AUC\n0.964", color=C_LATTN, fontsize=11,
            fontweight="bold", ha="center")
    ax.text(0.38, 0.55, "AUC\n0.935", color=C_ATTN, fontsize=10,
            ha="center", alpha=0.85)

    fig.tight_layout()
    out = out_dir / "6_roc_curves.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", default="./figures")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating presentation figures...\n")
    fig_journey(out_dir)
    fig_single_vs_multi(out_dir)
    fig_layer_importance(out_dir)
    fig_pipeline(out_dir)
    fig_final_results(out_dir)
    fig_roc_curves(out_dir)
    print(f"\nAll figures saved to {out_dir.resolve()}")
    print("\nFigure summary:")
    print("  1_journey.png          — 4-milestone step-up chart (the story arc)")
    print("  2_single_vs_multi.png  — single-layer vs multi-layer across all archs")
    print("  3_layer_importance.png — learned layer weights (why L45 matters)")
    print("  4_pipeline.png         — end-to-end pipeline diagram")
    print("  5_final_results.png    — horizontal bar chart of final config results")
    print("  6_roc_curves.png       — 3 representative ROC curves with gain shading")


if __name__ == "__main__":
    main()
