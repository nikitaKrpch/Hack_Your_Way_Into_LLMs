"""Visualize probe training results.

Usage:
    python visualize_probes.py \
        --results_dir ./probes/results \
        --out_dir ./probes/figures

Reads metrics.jsonl and generates:
- Bar chart: AUC comparison across architectures
- Heatmap: AUC by architecture x task
- Line plot: Training progression (if available)
- Box plot: Seed variance per architecture
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

sns.set_style("whitegrid")
PALETTE = sns.color_palette("husl", 8)


def load_metrics(results_dir):
    rows = []
    with open(results_dir / "metrics.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def aggregate_metrics(rows):
    """Group by (task, arch, regime) and compute mean/std across seeds."""
    stats = defaultdict(lambda: {"aucs": [], "accs": [], "f1s": []})
    for r in rows:
        key = (r["task"], r["arch"], r["regime"])
        stats[key]["aucs"].append(r["auc"])
        stats[key]["accs"].append(r["acc"])
        stats[key]["f1s"].append(r["f1"])

    agg = {}
    for key, vals in stats.items():
        agg[key] = {
            "auc_mean": np.nanmean(vals["aucs"]),
            "auc_std": np.nanstd(vals["aucs"]),
            "acc_mean": np.nanmean(vals["accs"]),
            "acc_std": np.nanstd(vals["accs"]),
            "f1_mean": np.nanmean(vals["f1s"]),
            "f1_std": np.nanstd(vals["f1s"]),
            "n_seeds": len(vals["aucs"]),
        }
    return agg


def plot_auc_comparison(agg, out_dir):
    """Bar chart: mean AUC by architecture, grouped by task."""
    tasks = sorted(set(k[0] for k in agg.keys()))
    archs = sorted(set(k[1] for k in agg.keys()))

    fig, ax = plt.subplots(figsize=(max(12, len(archs) * 2), 6))

    x = np.arange(len(archs))
    width = 0.8 / len(tasks)

    for i, task in enumerate(tasks):
        means = [agg[(task, a, "batch")]["auc_mean"] if (task, a, "batch") in agg
                else np.nan for a in archs]
        stds = [agg[(task, a, "batch")]["auc_std"] if (task, a, "batch") in agg
                else 0 for a in archs]
        offset = (i - len(tasks)/2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, label=task, yerr=stds, capsize=3,
                    color=PALETTE[i % len(PALETTE)], alpha=0.8)

    ax.set_xlabel("Architecture")
    ax.set_ylabel("AUC")
    ax.set_title("Probe AUC Comparison by Architecture")
    ax.set_xticks(x)
    ax.set_xticklabels(archs, rotation=45, ha="right")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(out_dir / "auc_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: {out_dir / 'auc_comparison.png'}")


def plot_auc_heatmap(agg, out_dir):
    """Heatmap: AUC by task x architecture."""
    tasks = sorted(set(k[0] for k in agg.keys()))
    archs = sorted(set(k[1] for k in agg.keys()))

    matrix = np.zeros((len(tasks), len(archs)))
    for i, task in enumerate(tasks):
        for j, arch in enumerate(archs):
            key = (task, arch, "batch")
            matrix[i, j] = agg.get(key, {}).get("auc_mean", np.nan)

    fig, ax = plt.subplots(figsize=(max(10, len(archs) * 1.2), max(6, len(tasks) * 0.8)))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0.5, vmax=1.0,
                xticklabels=archs, yticklabels=tasks, ax=ax, cbar_kws={"label": "AUC"})
    ax.set_title("Probe AUC Heatmap (task × architecture)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(out_dir / "auc_heatmap.png", dpi=150)
    plt.close()
    print(f"Saved: {out_dir / 'auc_heatmap.png'}")


def plot_regime_comparison(agg, out_dir):
    """Grouped bar chart: batch vs incremental per architecture."""
    archs = sorted(set(k[1] for k in agg.keys()))

    batch_means = [agg.get((t, a, "batch"), {}).get("auc_mean", np.nan)
                    for a in archs for t in set(k[0] for k in agg.keys())]
    batch_means = []
    inc_means = []
    labels = []
    for arch in archs:
        tasks_for_arch = sorted(set(k[0] for k in agg.keys() if k[1] == arch))
        for task in tasks_for_arch:
            b = agg.get((task, arch, "batch"), {}).get("auc_mean", np.nan)
            i = agg.get((task, arch, "incremental"), {}).get("auc_mean", np.nan)
            batch_means.append(b)
            inc_means.append(i)
            labels.append(f"{arch}\n({task[:15]}...)")

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(16, len(labels) * 0.8), 6))
    ax.bar(x - width/2, batch_means, width, label="batch", color="steelblue", alpha=0.8)
    ax.bar(x + width/2, inc_means, width, label="incremental", color="coral", alpha=0.8)

    ax.set_ylabel("AUC")
    ax.set_title("Batch vs Incremental Regime Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(out_dir / "regime_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: {out_dir / 'regime_comparison.png'}")


def plot_variance_by_arch(rows, out_dir):
    """Box plot: seed-level AUC variance per architecture."""
    arch_data = defaultdict(list)
    for r in rows:
        arch_data[r["arch"]].append(r["auc"])

    archs = sorted(arch_data.keys())
    data = [arch_data[a] for a in archs]

    fig, ax = plt.subplots(figsize=(max(10, len(archs) * 1.2), 6))
    bp = ax.boxplot(data, labels=archs, patch_artist=True)
    for patch, color in zip(bp['boxes'], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("AUC")
    ax.set_title("Seed-level AUC Variance by Architecture")
    ax.set_xticklabels(archs, rotation=45, ha="right")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(out_dir / "auc_variance.png", dpi=150)
    plt.close()
    print(f"Saved: {out_dir / 'auc_variance.png'}")


def plot_task_breakdown(agg, out_dir):
    """Multi-bar chart: breakdown by task type (refusal vs cyber_*)."""
    tasks = sorted(set(k[0] for k in agg.keys()))
    refusal_tasks = [t for t in tasks if t.startswith("refusal")]
    cyber_tasks = [t for t in tasks if t.startswith("cyber")]

    fig, axes = plt.subplots(1, 2, figsize=(max(14, len(cyber_tasks) * 2), 6))

    # Refusal tasks
    if refusal_tasks:
        archs = sorted(set(k[1] for k in agg.keys()))
        x = np.arange(len(refusal_tasks))
        width = 0.8 / len(archs)
        for i, arch in enumerate(archs):
            means = [agg.get((t, arch, "batch"), {}).get("auc_mean", np.nan) for t in refusal_tasks]
            stds = [agg.get((t, arch, "batch"), {}).get("auc_std", 0) for t in refusal_tasks]
            offset = (i - len(archs)/2 + 0.5) * width
            axes[0].bar(x + offset, means, width, label=arch, yerr=stds, capsize=2, alpha=0.8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([t.replace("refusal_", "") for t in refusal_tasks], rotation=45, ha="right")
        axes[0].set_title("Refusal Tasks")
        axes[0].set_ylabel("AUC")
        axes[0].legend(fontsize=7)
        axes[0].set_ylim(0, 1)

    # Cyber tasks
    if cyber_tasks:
        archs = sorted(set(k[1] for k in agg.keys()))
        x = np.arange(len(cyber_tasks))
        width = 0.8 / len(archs)
        for i, arch in enumerate(archs):
            means = [agg.get((t, arch, "batch"), {}).get("auc_mean", np.nan) for t in cyber_tasks]
            stds = [agg.get((t, arch, "batch"), {}).get("auc_std", 0) for t in cyber_tasks]
            offset = (i - len(archs)/2 + 0.5) * width
            axes[1].bar(x + offset, means, width, label=arch, yerr=stds, capsize=2, alpha=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([t.replace("cyber_", "").replace("_gemma4_31b", "") for t in cyber_tasks],
                                rotation=45, ha="right", fontsize=8)
        axes[1].set_title("Cyber Tasks")
        axes[1].legend(fontsize=7)
        axes[1].set_ylim(0, 1)

    plt.suptitle("AUC Breakdown by Task Type")
    plt.tight_layout()
    plt.savefig(out_dir / "task_breakdown.png", dpi=150)
    plt.close()
    print(f"Saved: {out_dir / 'task_breakdown.png'}")


def plot_summary_table(agg, out_dir):
    """Generate a summary table as image."""
    tasks = sorted(set(k[0] for k in agg.keys()))
    archs = sorted(set(k[1] for k in agg.keys()))

    # Compute overall mean AUC per architecture (across all tasks)
    arch_overall = {}
    for arch in archs:
        vals = [agg[(t, arch, "batch")]["auc_mean"] for t in tasks
                if (t, arch, "batch") in agg and not np.isnan(agg[(t, arch, "batch")]["auc_mean"])]
        arch_overall[arch] = (np.mean(vals), np.std(vals)) if vals else (np.nan, np.nan)

    # Sort by mean AUC
    sorted_archs = sorted(arch_overall.items(), key=lambda x: -x[1][0] if not np.isnan(x[1][0]) else -999)

    fig, ax = plt.subplots(figsize=(10, max(6, len(sorted_archs) * 0.5)))
    ax.axis("off")

    # Table data
    col_labels = ["Architecture", "Mean AUC", "Std AUC", "Rank"]
    cell_data = []
    for rank, (arch, (mean, std)) in enumerate(sorted_archs, 1):
        cell_data.append([arch, f"{mean:.4f}" if not np.isnan(mean) else "N/A",
                        f"{std:.4f}" if not np.isnan(std) else "N/A", str(rank)])

    table = ax.table(cellText=cell_data, colLabels=col_labels, loc="center",
                    cellLoc="center", colWidths=[0.3, 0.2, 0.2, 0.1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Color code best row
    for i, (arch, _) in enumerate(sorted_archs[:3], 0):
        if i < len(sorted_archs):
            table[(i+1, 0)].set_facecolor(PALETTE[i] + (0.3,))

    ax.set_title("Architecture Ranking by Mean AUC", pad=20, fontsize=14)

    plt.tight_layout()
    plt.savefig(out_dir / "summary_table.png", dpi=150)
    plt.close()
    print(f"Saved: {out_dir / 'summary_table.png'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", default="./probes/results",
                    help="Directory containing metrics.jsonl")
    ap.add_argument("--out_dir", default="./probes/figures",
                    help="Where to save figures")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading metrics from {results_dir}")
    rows = load_metrics(results_dir)
    print(f"Loaded {len(rows)} metric rows")

    if not rows:
        print("No data found in metrics.jsonl")
        return

    agg = aggregate_metrics(rows)

    print("\nGenerating visualizations...")
    plot_auc_comparison(agg, out_dir)
    plot_auc_heatmap(agg, out_dir)
    plot_regime_comparison(agg, out_dir)
    plot_variance_by_arch(rows, out_dir)
    plot_task_breakdown(agg, out_dir)
    plot_summary_table(agg, out_dir)

    print(f"\nAll figures saved to {out_dir}")


if __name__ == "__main__":
    main()

