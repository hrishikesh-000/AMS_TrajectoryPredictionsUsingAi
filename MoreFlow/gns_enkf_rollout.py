"""
gns_enkf_rollout.py

Wraps a trained GNS checkpoint in an ensemble Kalman filter (EnKF) and
compares it against a plain (uncorrected) rollout, evaluated against
DEM ground truth in <data_path>/test.npz.

All simulator-construction and prediction calls below are copied/adapted
directly from geoelements/gns's gns/train.py and gns/learned_simulator.py
(_get_simulator, predict_positions, LearnedSimulator.load) -- verified
against the actual GitHub source, not guessed.

Modes:
    --mode plain : single GNS rollout, no correction (the "before")
    --mode enkf  : n-member ensemble with periodic EnKF correction (the "after")
"""

import argparse
import os
import sys

import numpy as np
import torch

# gns_enkf_rollout.py lives outside the geoelements/gns repo checkout, so
# the repo root must be added to sys.path explicitly for `from gns import
# ...` to resolve -- gns/train.py does the same thing (relative to its own
# location) for this exact reason. Override with the GNS_REPO_ROOT env var
# if your clone isn't at ~/gns (e.g. a cluster with a different layout).
GNS_REPO_ROOT = os.environ.get("GNS_REPO_ROOT", os.path.expanduser("~/gns"))
if GNS_REPO_ROOT not in sys.path:
    sys.path.insert(0, GNS_REPO_ROOT)

from gns import learned_simulator
from gns import reading_utils

INPUT_SEQUENCE_LENGTH = 6
NUM_PARTICLE_TYPES = 9


def load_simulator(data_path, model_path, model_file, device):
    metadata = reading_utils.read_metadata(data_path, "rollout")

    normalization_stats = {
        'acceleration': {
            'mean': torch.FloatTensor(metadata['acc_mean']).to(device),
            'std': torch.sqrt(torch.FloatTensor(metadata['acc_std']) ** 2).to(device),
        },
        'velocity': {
            'mean': torch.FloatTensor(metadata['vel_mean']).to(device),
            'std': torch.sqrt(torch.FloatTensor(metadata['vel_std']) ** 2).to(device),
        },
    }

    if "nnode_in" in metadata and "nedge_in" in metadata:
        nnode_in = metadata['nnode_in']
        nedge_in = metadata['nedge_in']
    else:
        nnode_in = 37 if metadata['dim'] == 3 else 30
        nedge_in = metadata['dim'] + 1

    simulator = learned_simulator.LearnedSimulator(
        particle_dimensions=metadata['dim'],
        nnode_in=nnode_in,
        nedge_in=nedge_in,
        latent_dim=128,
        nmessage_passing_steps=10,
        nmlp_layers=2,
        mlp_hidden_dim=128,
        connectivity_radius=metadata['default_connectivity_radius'],
        boundaries=np.array(metadata['bounds']),
        normalization_stats=normalization_stats,
        nparticle_types=NUM_PARTICLE_TYPES,
        particle_type_embedding_size=16,
        boundary_clamp_limit=metadata.get("boundary_augment", 1.0),
        device=device,
    )

    simulator.load(os.path.join(model_path, model_file))
    simulator.to(device)
    simulator.eval()
    return simulator, metadata


def gns_predict_one_step(simulator, history, particle_type, n_particles, device):
    with torch.no_grad():
        next_position = simulator.predict_positions(
            history,
            nparticles_per_example=[n_particles],
            particle_types=particle_type,
            material_property=None,
        )
    return next_position


def make_history(positions_seq, device):
    arr = np.transpose(positions_seq, (1, 0, 2)).astype(np.float32)
    return torch.tensor(arr, device=device)


def slide_history(history, next_position):
    return torch.cat([history[:, 1:, :], next_position.unsqueeze(1)], dim=1)


def enkf_update(ensemble_positions, obs_position, obs_indices, obs_noise_std, ridge_eps):
    device = ensemble_positions.device
    n_members = ensemble_positions.shape[0]
    n_particles = ensemble_positions.shape[1]

    X = ensemble_positions.reshape(n_members, -1)
    x_mean = X.mean(dim=0, keepdim=True)
    A = (X - x_mean).T

    obs_cols = np.concatenate([obs_indices * 3 + d for d in range(3)])
    obs_cols_t = torch.tensor(obs_cols, device=device, dtype=torch.long)

    HA = A[obs_cols_t, :]
    y = torch.tensor(
        obs_position[obs_indices].reshape(-1), device=device, dtype=torch.float32
    )

    obs_dim = y.shape[0]
    R = (obs_noise_std ** 2) * torch.eye(obs_dim, device=device)

    denom = 1.0 / (n_members - 1)
    S = denom * (HA @ HA.T) + R + ridge_eps * torch.eye(obs_dim, device=device)
    # HA @ HA.T has rank <= n_members - 1 by construction (ensemble anomalies
    # sum to zero), so S can be exactly singular in the many directions the
    # ensemble carries no information about -- pinv handles that gracefully
    # via SVD, unlike inv which requires full rank.
    try:
        S_inv = torch.linalg.pinv(S)
    except torch._C._LinAlgError:
        # GPU cuSOLVER's SVD occasionally fails to converge on very
        # ill-conditioned matrices (worse the smaller obs_noise_std/ridge_eps
        # are, since that's less numerical padding away from true
        # singularity). CPU's LAPACK-based SVD is slower but far more
        # robust for exactly this edge case.
        S_inv = torch.linalg.pinv(S.cpu()).to(device)

    K_HA = denom * (A @ HA.T) @ S_inv

    corrected = torch.zeros_like(X)
    for j in range(n_members):
        y_j = y + obs_noise_std * torch.randn(obs_dim, device=device)
        innovation = y_j - X[j][obs_cols_t]
        corrected[j] = X[j] + K_HA @ innovation

    return corrected.reshape(n_members, n_particles, 3)


def run_rollout(args, device):
    ground_truth = np.load(
        os.path.join(args.data_path, "test.npz"), allow_pickle=True
    )["simulation_0"][0]
    n_particles = ground_truth.shape[1]
    n_frames = ground_truth.shape[0]

    simulator, metadata = load_simulator(
        args.data_path, args.model_path, args.model_file, device
    )
    particle_type = torch.zeros(n_particles, dtype=torch.long, device=device)

    # Physical safety clamp: on long/hard rollouts (extrapolating past what
    # the model trained on), predictions can occasionally diverge to huge or
    # non-finite values -- "particles flying out of frame". Since the real
    # container has fixed bounds, anything predicted far outside them is
    # obviously nonphysical. Clamping here (a) fixes the visible instability
    # and (b) stops NaN/Inf from ever reaching the EnKF covariance math,
    # which cannot handle non-finite input no matter which solver is used.
    bounds = np.array(metadata['bounds'], dtype=np.float32)  # shape (3, 2)
    span = bounds[:, 1] - bounds[:, 0]
    clamp_lo = torch.tensor(bounds[:, 0] - 0.5 * span, device=device).view(1, 1, 3)
    clamp_hi = torch.tensor(bounds[:, 1] + 0.5 * span, device=device).view(1, 1, 3)

    n_members = 1 if args.mode == "plain" else args.n_ensemble
    init_seq = ground_truth[:INPUT_SEQUENCE_LENGTH]

    ensemble_history = []
    for m in range(n_members):
        noise = 0.0 if m == 0 else np.random.normal(
            scale=args.init_perturb_std, size=init_seq.shape
        )
        ensemble_history.append(make_history(init_seq + noise, device))
    ensemble_history = torch.stack(ensemble_history)

    all_trajectories = torch.zeros(
        n_members, n_frames - INPUT_SEQUENCE_LENGTH, n_particles, 3, device=device
    )

    rng = np.random.default_rng(0)
    n_obs = max(1, int(round(args.obs_fraction * n_particles)))

    for step in range(INPUT_SEQUENCE_LENGTH, n_frames):
        idx = step - INPUT_SEQUENCE_LENGTH
        next_positions = torch.zeros(n_members, n_particles, 3, device=device)
        for m in range(n_members):
            next_positions[m] = gns_predict_one_step(
                simulator, ensemble_history[m], particle_type, n_particles, device
            )

        # Replace any NaN/Inf first (clamp alone doesn't fix NaN), then clamp
        # to the physical container bounds with margin.
        next_positions = torch.nan_to_num(next_positions, nan=0.0, posinf=0.0, neginf=0.0)
        next_positions = torch.clamp(next_positions, min=clamp_lo, max=clamp_hi)

        if args.mode == "enkf" and idx > 0 and idx % args.assim_every == 0:
            obs_indices = rng.choice(n_particles, size=n_obs, replace=False)
            try:
                next_positions = enkf_update(
                    next_positions,
                    ground_truth[step],
                    obs_indices,
                    args.obs_noise_std,
                    args.ridge_eps,
                )
                next_positions = torch.nan_to_num(next_positions, nan=0.0, posinf=0.0, neginf=0.0)
                next_positions = torch.clamp(next_positions, min=clamp_lo, max=clamp_hi)
            except torch._C._LinAlgError as e:
                # Correction failed for this frame (e.g. ensemble collapsed
                # to something SVD can't handle even after clamping) --
                # skip the correction for just this step rather than
                # crashing the whole rollout. The uncorrected (but already
                # clamped/finite) forecast is used instead for this frame.
                print(f"  [warn] skipping EnKF correction at step {idx}: {e}")

        all_trajectories[:, idx] = next_positions
        for m in range(n_members):
            ensemble_history[m] = slide_history(ensemble_history[m], next_positions[m])

        if idx % 20 == 0:
            print(f"step {idx}/{n_frames - INPUT_SEQUENCE_LENGTH}")

    out = all_trajectories.mean(dim=0).cpu().numpy()  # ensemble mean
    os.makedirs(args.output_path, exist_ok=True)
    np.save(os.path.join(args.output_path, f"{args.mode}_trajectories.npy"), out)
    np.save(
        os.path.join(args.output_path, "ground_truth.npy"),
        ground_truth[INPUT_SEQUENCE_LENGTH:],
    )
    print(f"Saved {args.mode} trajectories -> {args.output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--model_file", default="model-200000.pt")
    p.add_argument("--output_path", required=True)
    p.add_argument("--mode", choices=["plain", "enkf"], required=True)
    p.add_argument("--n_ensemble", type=int, default=30)
    p.add_argument("--assim_every", type=int, default=10)
    p.add_argument("--obs_fraction", type=float, default=1.0)
    p.add_argument("--obs_noise_std", type=float, default=1e-4)
    p.add_argument("--init_perturb_std", type=float, default=1e-4)
    p.add_argument("--ridge_eps", type=float, default=1e-8)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_rollout(args, device)
