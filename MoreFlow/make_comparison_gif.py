"""
make_enkf_comparison_gif.py

Renders a 3-panel side-by-side GIF from the outputs of gns_enkf_rollout.py:
    Reality (DEM ground truth) | Before (plain GNS) | After (GNS + EnKF)

Reads ground_truth.npy, plain_trajectories.npy, enkf_trajectories.npy from
--results_dir and saves enkf_comparison.gif there.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True, help="dir with ground_truth.npy, plain_trajectories.npy, enkf_trajectories.npy")
    p.add_argument("--out_dir", default=None, help="where to save the gif (default: --results_dir)")
    p.add_argument("--step_stride", type=int, default=2)
    p.add_argument("--fps", type=int, default=8)
    args = p.parse_args()
    if args.out_dir is None:
        args.out_dir = args.results_dir
    os.makedirs(args.out_dir, exist_ok=True)

    ground_truth = np.load(os.path.join(args.results_dir, "ground_truth.npy"))
    plain = np.load(os.path.join(args.results_dir, "plain_trajectories.npy"))
    enkf = np.load(os.path.join(args.results_dir, "enkf_trajectories.npy"))

    n_frames = ground_truth.shape[0]
    frame_indices = list(range(0, n_frames, args.step_stride))
    print(f"Rendering {len(frame_indices)} frames (every {args.step_stride}) out of {n_frames}")

    fig = plt.figure(figsize=(15, 5))
    axes = [fig.add_subplot(1, 3, i + 1, projection='3d') for i in range(3)]
    base_titles = ["Reality (DEM)", "Before (plain GNS)", "After (GNS + EnKF)"]
    datasets = [ground_truth, plain, enkf]

    def frame_mse(data, fi):
        if data is ground_truth:
            return None
        err = data[fi] - ground_truth[fi]
        return float(np.mean(err ** 2))

    def update(frame_pos):
        fi = frame_indices[frame_pos]
        for ax, data, base_title in zip(axes, datasets, base_titles):
            ax.cla()
            pos = data[fi]
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=1, c='green', alpha=0.6)
            ax.set_xlim(-0.05, 0.05)
            ax.set_ylim(-0.05, 0.05)
            ax.set_zlim(-0.08, 0.02)
            mse = frame_mse(data, fi)
            title = base_title if mse is None else f"{base_title}\nMSE: {mse:.2e}"
            ax.set_title(title, fontsize=10)
        fig.suptitle(f"Frame {fi}/{n_frames}", fontsize=11)

    anim = FuncAnimation(fig, update, frames=len(frame_indices), interval=100)
    out_path = os.path.join(args.out_dir, "enkf_comparison.gif")
    anim.save(out_path, writer=PillowWriter(fps=args.fps))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
