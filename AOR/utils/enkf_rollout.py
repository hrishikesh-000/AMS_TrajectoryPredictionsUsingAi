"""Ensemble Kalman Filter (EnKF) correction layer on top of a trained GNS rollout.

Runs an N-member ensemble of perturbed GNS rollouts side by side. At chosen
checkpoint steps, blends each member's prediction toward a (noised) copy of
the true position, weighted by a Kalman gain computed from the ensemble's own
spread (diagonal/localized covariance - independent per particle, per axis;
the only tractable choice when the ensemble is far smaller than the state
dimension, e.g. 30 members vs 2000 particles x 3 dims = 6000 state dims).

Reuses the exact simulator construction (_get_simulator) from gns/train.py so
the model is built identically to how it was trained.

Usage:
  python enkf_rollout.py \
      --data_dir=C:/tmp/aor_2000p_honest_split \
      --model_path=C:/tmp/models_aor_2000p_honest/model-200000.pt \
      --out_dir=C:/tmp/rollouts_aor_2000p_enkf \
      --n_ensemble=30 --correction_interval=5 --obs_noise_std=1e-4

  Add --disable_correction to run the identical ensemble pipeline with the
  Kalman update skipped - a control run isolating the effect of correction
  itself, rather than comparing against the earlier single-deterministic-run
  baseline (which didn't use ensembling at all).
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gns.train import _get_simulator, INPUT_SEQUENCE_LENGTH, NUM_PARTICLE_TYPES  # noqa: E402
from gns.data_loader import load_npz_data  # noqa: E402

# Some platforms' torch_geometric build requires pyg-lib for radius_graph
# (no Windows wheel for pyg-lib exists, so this fails there). Where that's the
# case, fall back to torch_cluster.radius_graph, which has an identical
# signature. On platforms where pyg-lib is already present (e.g. the cluster's
# Linux env), this is skipped entirely - no behavior change there.
import torch_geometric  # noqa: E402
# Older torch_geometric versions (e.g. on the cluster) have no WITH_RADIUS
# attribute at all - they used torch_cluster internally with no such gate, so
# default to "already works, no patch needed" when the attribute is absent.
if not getattr(torch_geometric.typing, "WITH_RADIUS", True):
    import torch_cluster  # noqa: E402
    from gns import learned_simulator as _ls_module  # noqa: E402
    _ls_module.radius_graph = torch_cluster.radius_graph
    print("Note: patched radius_graph -> torch_cluster (pyg-lib unavailable on this platform)")


def load_trajectory(npz_path):
    data = load_npz_data(npz_path)
    positions, particle_type = data[0]  # (n_frames, n_particles, dim), scalar type
    positions = np.transpose(positions, (1, 0, 2))  # -> (n_particles, n_frames, dim)
    return positions.astype(np.float32), particle_type


def build_ensemble_state(seed_positions, n_ensemble, seed_noise_std, device):
    # seed_positions: (n_particles, INPUT_SEQUENCE_LENGTH, dim)
    n_particles, seq_len, dim = seed_positions.shape
    members = []
    for i in range(n_ensemble):
        noise = np.random.normal(0, seed_noise_std, size=seed_positions.shape) if i > 0 else 0.0
        members.append(seed_positions + noise)
    stacked = np.stack(members, axis=0)  # (n_ensemble, n_particles, seq_len, dim)
    flat = stacked.reshape(n_ensemble * n_particles, seq_len, dim)
    return torch.tensor(flat, dtype=torch.float32, device=device)


def enkf_correct(ensemble_next_pos, observation, obs_noise_std):
    # ensemble_next_pos: (n_ensemble, n_particles, dim) numpy
    # observation: (n_particles, dim) numpy - true position at this checkpoint
    n_ensemble = ensemble_next_pos.shape[0]
    ensemble_mean = ensemble_next_pos.mean(axis=0)  # (n_particles, dim)
    ensemble_var = ensemble_next_pos.var(axis=0)  # (n_particles, dim), diagonal/localized covariance
    kalman_gain = ensemble_var / (ensemble_var + obs_noise_std ** 2)  # (n_particles, dim)

    corrected = np.empty_like(ensemble_next_pos)
    for i in range(n_ensemble):
        perturbed_obs = observation + np.random.normal(0, obs_noise_std, size=observation.shape)
        corrected[i] = ensemble_next_pos[i] + kalman_gain * (perturbed_obs - ensemble_next_pos[i])
    return corrected, kalman_gain


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        metadata = json.load(f)

    simulator = _get_simulator(metadata, args.noise_std, args.noise_std, device)
    simulator.load(args.model_path)
    simulator.to(device)
    simulator.eval()

    positions, particle_type_scalar = load_trajectory(os.path.join(args.data_dir, args.split_file))
    n_particles = positions.shape[0]
    n_frames = positions.shape[1]
    nsteps = n_frames - INPUT_SEQUENCE_LENGTH
    print(f"Loaded {args.split_file} trajectory: {n_particles} particles, {n_frames} frames, {nsteps} rollout steps")

    seed = positions[:, :INPUT_SEQUENCE_LENGTH, :]  # (n_particles, 6, dim)
    ground_truth = positions[:, INPUT_SEQUENCE_LENGTH:, :]  # (n_particles, nsteps, dim)

    current = build_ensemble_state(seed, args.n_ensemble, args.seed_noise_std, device)
    particle_types = torch.full((args.n_ensemble * n_particles,), int(particle_type_scalar),
                                 dtype=torch.long, device=device)
    nparticles_per_example = [n_particles] * args.n_ensemble

    predicted_mean_trajectory = []
    kalman_gains_at_checkpoints = []

    with torch.no_grad():
        for step in range(nsteps):
            next_flat = simulator.predict_positions(
                current, nparticles_per_example=nparticles_per_example,
                particle_types=particle_types)  # (n_ensemble*n_particles, dim)
            next_np = next_flat.cpu().numpy().reshape(args.n_ensemble, n_particles, -1)

            is_checkpoint = (not args.disable_correction) and ((step + 1) % args.correction_interval == 0)
            if is_checkpoint:
                observation = ground_truth[:, step, :]  # true position at this rollout step
                next_np, kgain = enkf_correct(next_np, observation, args.obs_noise_std)
                kalman_gains_at_checkpoints.append((step + 1, float(kgain.mean())))

            predicted_mean_trajectory.append(next_np.mean(axis=0))

            next_flat_corrected = torch.tensor(
                next_np.reshape(args.n_ensemble * n_particles, -1), dtype=torch.float32, device=device)
            current = torch.cat([current[:, 1:, :], next_flat_corrected[:, None, :]], dim=1)

            if (step + 1) % 5 == 0 or step == nsteps - 1:
                print(f"step {step + 1}/{nsteps} done"
                      + (" [CORRECTED]" if is_checkpoint else ""))

    predicted = np.stack(predicted_mean_trajectory, axis=0)  # (nsteps, n_particles, dim)
    gt = np.transpose(ground_truth, (1, 0, 2))  # (nsteps, n_particles, dim)

    rmse = float(np.sqrt(np.mean((predicted - gt) ** 2)))
    mae = float(np.mean(np.abs(predicted - gt)))
    ss_res = np.sum((predicted - gt) ** 2)
    ss_tot = np.sum((gt - gt.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot)
    per_step_rmse = np.sqrt(np.mean((predicted - gt) ** 2, axis=(1, 2)))

    print("\n" + "=" * 40)
    print("EnKF ROLLOUT METRICS" if not args.disable_correction else "CONTROL (no correction) METRICS")
    print("=" * 40)
    print(f"RMSE: {rmse:.6e} m")
    print(f"MAE:  {mae:.6e} m")
    print(f"R2:   {r2:.6f}")
    if kalman_gains_at_checkpoints:
        print("Mean Kalman gain at each checkpoint step:")
        for step_num, kg in kalman_gains_at_checkpoints:
            print(f"  step {step_num}: {kg:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_name = "rollout_enkf.pkl" if not args.disable_correction else "rollout_control_no_correction.pkl"
    with open(os.path.join(args.out_dir, out_name), "wb") as f:
        pickle.dump({
            "predicted_rollout": predicted,
            "ground_truth_rollout": gt,
            "per_step_rmse": per_step_rmse,
            "rmse": rmse, "mae": mae, "r2": r2,
            "n_ensemble": args.n_ensemble,
            "correction_interval": args.correction_interval,
            "obs_noise_std": args.obs_noise_std,
            "disable_correction": args.disable_correction,
        }, f)
    print(f"Saved to {os.path.join(args.out_dir, out_name)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="Folder with test.npz + metadata.json")
    p.add_argument("--split_file", default="test.npz", help="Which npz in data_dir to roll out on (e.g. train.npz)")
    p.add_argument("--model_path", required=True, help="Path to trained model .pt file")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_ensemble", type=int, default=30)
    p.add_argument("--correction_interval", type=int, default=5)
    p.add_argument("--obs_noise_std", type=float, default=1e-4,
                    help="Assumed noise level of the 'observation' used for correction (meters)")
    p.add_argument("--seed_noise_std", type=float, default=3e-4,
                    help="Perturbation applied to seed frames to create initial ensemble spread")
    p.add_argument("--noise_std", type=float, default=3e-4,
                    help="Must match the noise_std the model was trained with (affects normalization stats)")
    p.add_argument("--disable_correction", action="store_true",
                    help="Run the same ensemble pipeline but skip the Kalman update (control/ablation run)")
    args = p.parse_args()
    run(args)
