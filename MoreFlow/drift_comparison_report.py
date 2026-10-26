"""
drift_comparison_report.py

Turns a plain-vs-EnKF rollout result (ground_truth.npy, plain_trajectories.npy,
enkf_trajectories.npy in --results_dir) into:
    drift_comparison_table.csv  -- per-frame RMSE before/after + % improvement
    drift_comparison_table.txt  -- same thing, human-readable
    drift_comparison_plot.png   -- RMSE-vs-frame line plot (drift growth,
                                    before vs after) + an overall bar chart

Works on any results_dir produced by a gns_enkf_rollout.py-style script,
regardless of which dataset/checkpoint it came from.
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics_utils import compute_all_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True,
                    help="dir with ground_truth.npy, plain_trajectories.npy, enkf_trajectories.npy")
    p.add_argument("--out_dir", default=None, help="default: --results_dir")
    args = p.parse_args()
    if args.out_dir is None:
        args.out_dir = args.results_dir
    os.makedirs(args.out_dir, exist_ok=True)

    ground_truth = np.load(os.path.join(args.results_dir, "ground_truth.npy"))
    plain = np.load(os.path.join(args.results_dir, "plain_trajectories.npy"))
    enkf = np.load(os.path.join(args.results_dir, "enkf_trajectories.npy"))

    before = compute_all_metrics(plain, ground_truth)
    after = compute_all_metrics(enkf, ground_truth)

    n_frames = ground_truth.shape[0]
    rmse_before = before["per_frame"]["rmse"]
    rmse_after = after["per_frame"]["rmse"]

    # ---- table ----
    csv_path = os.path.join(args.out_dir, "drift_comparison_table.csv")
    txt_path = os.path.join(args.out_dir, "drift_comparison_table.txt")
    rows = []
    for t in range(n_frames):
        b, a = rmse_before[t], rmse_after[t]
        pct = 100.0 * (b - a) / b if b != 0 else float("nan")
        rows.append((t, b, a, pct))

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "rmse_before_enkf", "rmse_after_enkf", "improvement_pct"])
        for t, b, a, pct in rows:
            w.writerow([t, f"{b:.6e}", f"{a:.6e}", f"{pct:.3f}"])
    print(f"Saved -> {csv_path}")

    with open(txt_path, "w") as f:
        f.write("Drift comparison: before (plain GNS) vs after (EnKF)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Frame':>6} | {'RMSE before':>14} | {'RMSE after':>14} | {'Improvement':>12}\n")
        f.write("-" * 60 + "\n")
        # sample every 10th frame for the readable table (full detail is in the .csv)
        for t, b, a, pct in rows[::10]:
            f.write(f"{t:>6} | {b:>14.6e} | {a:>14.6e} | {pct:>11.2f}%\n")
        f.write("-" * 60 + "\n\n")
        f.write("Overall (all frames):\n")
        f.write(f"  RMSE before EnKF: {before['overall']['rmse']:.6e}\n")
        f.write(f"  RMSE after EnKF:  {after['overall']['rmse']:.6e}\n")
        overall_pct = 100.0 * (before['overall']['rmse'] - after['overall']['rmse']) / before['overall']['rmse']
        f.write(f"  Overall RMSE improvement: {overall_pct:.2f}%\n\n")
        f.write(f"  MSE before / after: {before['overall']['mse']:.6e} / {after['overall']['mse']:.6e}\n")
        f.write(f"  R2   before / after: {before['overall']['r2']:.6f} / {after['overall']['r2']:.6f}\n")
    print(f"Saved -> {txt_path}")

    # ---- plot ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(range(n_frames), rmse_before, label="Before (plain GNS)", color="#e74c3c")
    axes[0].plot(range(n_frames), rmse_after, label="After (EnKF)", color="#2ecc71")
    axes[0].set_xlabel("Rollout frame")
    axes[0].set_ylabel("RMSE (drift from ground truth)")
    axes[0].set_title("Drift growth over rollout: before vs after EnKF")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    drift_reduction = np.array(rmse_before) - np.array(rmse_after)
    axes[1].plot(range(n_frames), drift_reduction, color="#3498db")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].fill_between(range(n_frames), drift_reduction, 0, alpha=0.2, color="#3498db")
    axes[1].set_xlabel("Rollout frame")
    axes[1].set_ylabel("RMSE reduction (before - after)")
    axes[1].set_title("How much EnKF reduced drift, per frame")
    axes[1].grid(alpha=0.3)

    labels = ["RMSE", "MSE"]
    before_vals = [before["overall"]["rmse"], before["overall"]["mse"]]
    after_vals = [after["overall"]["rmse"], after["overall"]["mse"]]
    x = np.arange(len(labels))
    width = 0.35
    axes[2].bar(x - width / 2, before_vals, width, label="Before", color="#e74c3c")
    axes[2].bar(x + width / 2, after_vals, width, label="After", color="#2ecc71")
    axes[2].set_yscale("log")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_title("Overall error: before vs after EnKF")
    axes[2].legend()
    axes[2].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = os.path.join(args.out_dir, "drift_comparison_plot.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved -> {plot_path}")


if __name__ == "__main__":
    main()
