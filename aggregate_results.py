"""Aggregate JSON results from results/ directory.

Usage:
    python aggregate_results.py [--results_dir results] [--wandb_project PROJECT]
"""

import argparse
import json
import os
import glob
from collections import defaultdict

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Aggregate MaNo experiment results.")
    parser.add_argument("--results_dir", default="results", type=str)
    parser.add_argument("--wandb_project", default=None, type=str,
                        help="If set, upload summary table to wandb.")
    args = parser.parse_args()

    json_files = sorted(glob.glob(os.path.join(args.results_dir, "**", "*.json"), recursive=True))
    if not json_files:
        print("No JSON result files found in '{}'.".format(args.results_dir))
        return

    # Collect per-arch metrics
    arch_metrics = defaultdict(lambda: {"R2": [], "spearman": []})
    all_r2 = []
    all_sp = []
    rows = []

    for fpath in json_files:
        with open(fpath, "r") as f:
            data = json.load(f)
        arch = data.get("arch", "unknown")
        dataname = data.get("dataname", "unknown")
        alg = data.get("alg", "unknown")
        r2 = data.get("R2", None)
        sp = data.get("spearman", None)
        if r2 is None or sp is None:
            print("  [skip] {} (missing R2 or spearman)".format(fpath))
            continue

        arch_metrics[arch]["R2"].append(r2)
        arch_metrics[arch]["spearman"].append(sp)
        all_r2.append(r2)
        all_sp.append(sp)
        rows.append({
            "file": os.path.relpath(fpath, args.results_dir),
            "alg": alg, "dataname": dataname, "arch": arch,
            "R2": r2, "spearman": sp,
        })

    # Print per-file
    print("=" * 70)
    print("Individual results ({} files)".format(len(rows)))
    print("=" * 70)
    for row in rows:
        print("  {file:50s}  R2={R2:.4f}  Spearman={spearman:.4f}".format(**row))

    # Print per-arch averages
    print("\n" + "=" * 70)
    print("Per-architecture averages")
    print("=" * 70)
    for arch in sorted(arch_metrics.keys()):
        m = arch_metrics[arch]
        print("  {:20s}  R2={:.4f}  Spearman={:.4f}  (n={})".format(
            arch, np.mean(m["R2"]), np.mean(m["spearman"]), len(m["R2"])))

    # Print overall average
    print("\n" + "=" * 70)
    print("Overall average (n={})".format(len(all_r2)))
    print("=" * 70)
    print("  R2={:.4f}  Spearman={:.4f}".format(np.mean(all_r2), np.mean(all_sp)))

    # Optional wandb upload
    if args.wandb_project:
        try:
            import wandb
            run = wandb.init(project=args.wandb_project, job_type="aggregate")
            table = wandb.Table(columns=["file", "alg", "dataname", "arch", "R2", "spearman"])
            for row in rows:
                table.add_data(row["file"], row["alg"], row["dataname"],
                               row["arch"], row["R2"], row["spearman"])
            run.log({"results_table": table})
            run.summary["overall_R2"] = float(np.mean(all_r2))
            run.summary["overall_spearman"] = float(np.mean(all_sp))
            run.finish()
            print("\n[aggregate] Uploaded summary table to wandb project '{}'.".format(args.wandb_project))
        except ImportError:
            print("\n[aggregate] wandb not installed, skipping upload.")


if __name__ == "__main__":
    main()
