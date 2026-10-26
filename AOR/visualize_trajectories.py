import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import sys
import os

def load_rollout(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    pred = np.array(data['predicted_rollout'])
    gt   = np.array(data['ground_truth_rollout'])
    return pred, gt

def visualize_trajectories(pkl_path, n_particles=50, output_gif=None):
    pred, gt = load_rollout(pkl_path)
    T, N, D = pred.shape
    print(f"Rollout: {T} steps, {N} particles, {D} dims")

    # Pick n_particles random particles to show trajectories for
    np.random.seed(42)
    idx = np.random.choice(N, min(n_particles, N), replace=False)

    # Compute per-particle total displacement to find most-moving particles
    displacement = np.linalg.norm(gt[-1] - gt[0], axis=1)
    # Pick top movers + some random ones
    top_movers = np.argsort(displacement)[-n_particles//2:]
    random_idx = np.random.choice(N, n_particles//2, replace=False)
    idx = np.unique(np.concatenate([top_movers, random_idx]))[:n_particles]

    print(f"Tracking {len(idx)} particles (top movers + random)")
    print(f"Max displacement: {displacement.max()*1000:.2f} mm")
    print(f"Mean displacement: {displacement.mean()*1000:.2f} mm")

    # Color particles by their total displacement
    displacements = displacement[idx]
    colors = plt.cm.plasma(
        (displacements - displacements.min()) /
        (displacements.max() - displacements.min() + 1e-10)
    )

    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor('#0d0d0d')

    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    for ax, title in [(ax1, 'Ground Truth (DEM)'), (ax2, 'GNS Prediction')]:
        ax.set_facecolor('#0d0d0d')
        ax.set_title(title, color='white', fontsize=11, pad=8)
        ax.tick_params(colors='gray', labelsize=7)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#333333')
        ax.yaxis.pane.set_edgecolor('#333333')
        ax.zaxis.pane.set_edgecolor('#333333')
        ax.grid(True, color='#222222', linewidth=0.5)

    # Compute axis limits from all positions
    all_pos = np.concatenate([gt[:, idx, :], pred[:, idx, :]], axis=0)
    margin = 0.005
    xlim = (all_pos[:,:,0].min()-margin, all_pos[:,:,0].max()+margin)
    ylim = (all_pos[:,:,1].min()-margin, all_pos[:,:,1].max()+margin)
    zlim = (all_pos[:,:,2].min()-margin, all_pos[:,:,2].max()+margin)

    trail_length = 15  # how many past steps to show as trail

    def update(frame):
        ax1.cla()
        ax2.cla()

        for ax, title in [(ax1, f'Ground Truth (DEM)\nStep {frame+1}/{T}'),
                          (ax2, f'GNS Prediction\nStep {frame+1}/{T}')]:
            ax.set_facecolor('#0d0d0d')
            ax.set_title(title, color='white', fontsize=10, pad=6)
            ax.tick_params(colors='gray', labelsize=6)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('#333333')
            ax.yaxis.pane.set_edgecolor('#333333')
            ax.zaxis.pane.set_edgecolor('#333333')
            ax.grid(True, color='#222222', linewidth=0.5)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
            ax.set_xlabel('X', color='gray', fontsize=7)
            ax.set_ylabel('Y', color='gray', fontsize=7)
            ax.set_zlabel('Z', color='gray', fontsize=7)

        for i, (pid, color) in enumerate(zip(idx, colors)):
            # Trail
            start = max(0, frame - trail_length)
            trail_frames = range(start, frame + 1)

            gt_trail   = gt[list(trail_frames), pid, :]
            pred_trail = pred[list(trail_frames), pid, :]

            alpha_vals = np.linspace(0.1, 0.9, len(trail_frames))

            # Draw trail lines
            for t in range(len(trail_frames) - 1):
                a = alpha_vals[t]
                ax1.plot(gt_trail[t:t+2, 0], gt_trail[t:t+2, 1],
                         gt_trail[t:t+2, 2], color=color, alpha=a, linewidth=0.8)
                ax2.plot(pred_trail[t:t+2, 0], pred_trail[t:t+2, 1],
                         pred_trail[t:t+2, 2], color=color, alpha=a, linewidth=0.8)

            # Current position dot
            ax1.scatter(*gt[frame, pid, :], color=color, s=12, alpha=1.0, depthshade=False)
            ax2.scatter(*pred[frame, pid, :], color=color, s=12, alpha=1.0, depthshade=False)

        # RMSE at this frame
        rmse_t = np.sqrt(np.mean((pred[frame] - gt[frame])**2))
        fig.suptitle(f'Step {frame+1}/{T}  |  RMSE: {rmse_t*1000:.3f} mm',
                     color='white', fontsize=11, y=1.0)

        return []

    print("Generating animation...")
    ani = animation.FuncAnimation(fig, update, frames=T,
                                  interval=100, blit=False)

    if output_gif:
        print(f"Saving to {output_gif}...")
        ani.save(output_gif, writer='pillow', fps=10,
                 savefig_kwargs={'facecolor': '#0d0d0d'})
        print(f"Saved: {output_gif}")
    else:
        plt.show()

    plt.close()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python visualize_trajectories.py <rollout_pkl> [output.gif] [n_particles]")
        sys.exit(1)

    pkl_path = sys.argv[1]
    output_gif = sys.argv[2] if len(sys.argv) > 2 else None
    n_particles = int(sys.argv[3]) if len(sys.argv) > 3 else 50

    visualize_trajectories(pkl_path, n_particles=n_particles, output_gif=output_gif)
