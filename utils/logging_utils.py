"""Logging utilities for wandb integration, scatter plots, and JSON result saving."""

import os
import json
import numpy as np


def init_wandb(args, project=None, group=None, tags=None):
    """Initialize a wandb run. Returns None if wandb is not installed or project is not set."""
    if not project:
        return None
    try:
        import wandb
    except ImportError:
        print("[logging_utils] wandb not installed, skipping wandb logging.")
        return None

    config = dict(args) if isinstance(args, dict) else vars(args)
    run = wandb.init(
        project=project,
        group=group,
        tags=tags,
        config=config,
        name="{}_{}_{}_{}".format(
            config.get("alg", ""),
            config.get("dataname", ""),
            config.get("arch", ""),
            config.get("seed", ""),
        ),
    )
    return run


def log_iteration(wandb_run, iter_idx, corruption, severity, score, test_acc, iter_time, source=None):
    """Log a single iteration to wandb."""
    if wandb_run is None:
        return
    payload = {
        "iter": iter_idx,
        "corruption": corruption,
        "severity": severity,
        "score": score,
        "test_acc": test_acc,
        "iter_time": iter_time,
    }
    if source is not None:
        payload["source"] = source
    wandb_run.log(payload)


def log_summary(wandb_run, r2, spearman_corr, mean_score, mean_time, total_wall):
    """Write final summary metrics to wandb."""
    if wandb_run is None:
        return
    wandb_run.summary["R2"] = r2
    wandb_run.summary["spearman"] = spearman_corr
    wandb_run.summary["mean_score"] = mean_score
    wandb_run.summary["mean_time"] = mean_time
    wandb_run.summary["total_wall_time"] = total_wall


def make_scatter_plot(scores, accs, color_labels, title, save_path):
    """Create a scatter plot with regression line and R2/Spearman annotation.

    Args:
        scores: list of MaNo scores (x-axis).
        accs: list of test accuracies (y-axis).
        color_labels: list of category strings (corruption or source) for coloring.
        title: plot title.
        save_path: output file path (PDF recommended).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    unique_labels = sorted(set(color_labels))
    n_colors = len(unique_labels)
    if n_colors <= 10:
        cmap = plt.cm.tab10
    else:
        cmap = plt.cm.tab20
    label_to_color = {label: cmap(i / max(n_colors - 1, 1)) for i, label in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(8, 6))

    for label in unique_labels:
        idx = [i for i, l in enumerate(color_labels) if l == label]
        xs = [scores[i] for i in idx]
        ys = [accs[i] for i in idx]
        ax.scatter(xs, ys, color=label_to_color[label], label=label, alpha=0.7, edgecolors="k", linewidths=0.3, s=40)

    # Regression line
    scores_arr = np.array(scores)
    accs_arr = np.array(accs)
    if len(scores_arr) > 1:
        slope, intercept = np.polyfit(scores_arr, accs_arr, 1)
        x_line = np.linspace(scores_arr.min(), scores_arr.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, "--", color="gray", linewidth=1.5)

    # R2 and Spearman
    r2 = float(np.corrcoef(scores_arr, accs_arr)[0, 1] ** 2) if len(scores_arr) > 1 else 0.0
    sp = float(stats.spearmanr(scores_arr, accs_arr).correlation) if len(scores_arr) > 1 else 0.0
    textstr = "R\u00b2 = {:.4f}\nSpearman = {:.4f}".format(r2, sp)
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment="bottom", bbox=props)

    ax.set_xlabel("MaNo Score")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7, ncol=max(1, n_colors // 10))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print("[logging_utils] Scatter plot saved to {}".format(save_path))


def log_scatter_to_wandb(wandb_run, save_path):
    """Upload a saved scatter plot to wandb as an image."""
    if wandb_run is None:
        return
    try:
        import wandb
        wandb_run.log({"scatter_plot": wandb.Image(save_path)})
    except Exception as e:
        print("[logging_utils] Failed to log scatter to wandb: {}".format(e))


def save_results_json(results_dict, save_path):
    """Save results dictionary to a JSON file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print("[logging_utils] Results saved to {}".format(save_path))


def finish_wandb(wandb_run):
    """Finish the wandb run."""
    if wandb_run is None:
        return
    wandb_run.finish()
