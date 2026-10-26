"""Convert LIGGGHTS DEM CSV output (all_data_01_AOR / all_data_02_more_Flow style)
into GNS-format train.npz / test.npz / metadata.json.

Builds a fixed-particle-count trajectory by tracking the first N particle IDs
(by ID, i.e. the earliest-injected particles) across a contiguous frame range,
then splits that range into a train window and a held-out (still dynamic) test
window - avoiding the settled-phase test-split issue described in CONTEXT.md.

Usage:
  python convert_aor_csv_to_npz.py \
      --csv_dir=all_data_01_AOR \
      --n_particles=2000 \
      --train_start=21 --train_end=160 \
      --test_start=161 --test_end=200 \
      --particle_type=5 \
      --connectivity_radius=0.006 \
      --dt=0.0001 \
      --out_dir=C:/tmp/aor_2000p_honest_split
"""
import argparse
import csv
import json
import os

import numpy as np


def read_frame(csv_dir, file_prefix, frame, particle_ids):
    path = os.path.join(csv_dir, f"{file_prefix}_{frame}.csv")
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = {int(r["id"]): r for r in reader if int(r["id"]) in particle_ids}
    missing = particle_ids - rows.keys()
    if missing:
        raise ValueError(f"Frame {frame}: missing {len(missing)} particle ids, "
                          f"e.g. {sorted(missing)[:5]}")
    positions = np.empty((len(particle_ids), 3), dtype=np.float64)
    for i, pid in enumerate(sorted(particle_ids)):
        r = rows[pid]
        positions[i] = [float(r["Points:0"]), float(r["Points:1"]), float(r["Points:2"])]
    return positions


def build_trajectory(csv_dir, file_prefix, start, end, particle_ids):
    frames = []
    for frame in range(start, end + 1):
        frames.append(read_frame(csv_dir, file_prefix, frame, particle_ids))
    return np.stack(frames, axis=0)  # (n_frames, n_particles, 3)


def compute_vel_acc_stats(positions):
    # positions: (n_frames, n_particles, 3). Matches learned_simulator.py convention:
    # velocity = pos[t] - pos[t-1], acceleration = vel[t] - vel[t-1] (raw frame deltas, no dt scaling).
    velocity = np.diff(positions, axis=0)
    acceleration = np.diff(velocity, axis=0)
    vel_mean = velocity.reshape(-1, 3).mean(axis=0)
    vel_std = velocity.reshape(-1, 3).std(axis=0)
    acc_mean = acceleration.reshape(-1, 3).mean(axis=0)
    acc_std = acceleration.reshape(-1, 3).std(axis=0)
    return vel_mean, vel_std, acc_mean, acc_std


def compute_global_bounds(csv_dir, file_prefix, n_files):
    mins = np.array([np.inf, np.inf, np.inf])
    maxs = np.array([-np.inf, -np.inf, -np.inf])
    for frame in range(1, n_files + 1):
        path = os.path.join(csv_dir, f"{file_prefix}_{frame}.csv")
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            pts = np.array([[float(r["Points:0"]), float(r["Points:1"]), float(r["Points:2"])]
                             for r in reader])
        mins = np.minimum(mins, pts.min(axis=0))
        maxs = np.maximum(maxs, pts.max(axis=0))
    return mins, maxs


def find_file_prefix(csv_dir):
    for fname in sorted(os.listdir(csv_dir)):
        if fname.endswith("_1.csv"):
            return fname[: -len("_1.csv")]
    raise FileNotFoundError(f"Could not find a '*_1.csv' file in {csv_dir}")


def save_gns_npz(path, positions, particle_type):
    arr = np.empty(1, dtype=object)
    arr[0] = (positions, particle_type)
    np.savez(path, gns_data=arr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_dir", required=True)
    p.add_argument("--n_particles", type=int, required=True)
    p.add_argument("--train_start", type=int, required=True)
    p.add_argument("--train_end", type=int, required=True)
    p.add_argument("--test_start", type=int, required=True)
    p.add_argument("--test_end", type=int, required=True)
    p.add_argument("--particle_type", type=int, default=5)
    p.add_argument("--connectivity_radius", type=float, default=0.006)
    p.add_argument("--dt", type=float, default=0.0001)
    p.add_argument("--n_files_for_bounds", type=int, default=400)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    file_prefix = find_file_prefix(args.csv_dir)
    particle_ids = set(range(1, args.n_particles + 1))

    print(f"Building train trajectory: frames {args.train_start}-{args.train_end}, "
          f"{args.n_particles} particles")
    train_positions = build_trajectory(args.csv_dir, file_prefix,
                                        args.train_start, args.train_end, particle_ids)

    print(f"Building test trajectory: frames {args.test_start}-{args.test_end}, "
          f"{args.n_particles} particles")
    test_positions = build_trajectory(args.csv_dir, file_prefix,
                                       args.test_start, args.test_end, particle_ids)

    print("Computing velocity/acceleration normalization stats from TRAIN split only")
    vel_mean, vel_std, acc_mean, acc_std = compute_vel_acc_stats(train_positions)

    print(f"Scanning all {args.n_files_for_bounds} frames for domain bounds (full particle set)")
    mins, maxs = compute_global_bounds(args.csv_dir, file_prefix, args.n_files_for_bounds)

    metadata = {
        "bounds": [[float(mins[d]), float(maxs[d])] for d in range(3)],
        "sequence_length": int(test_positions.shape[0]),
        "default_connectivity_radius": args.connectivity_radius,
        "dim": 3,
        "dt": args.dt,
        "vel_mean": vel_mean.tolist(),
        "vel_std": vel_std.tolist(),
        "acc_mean": acc_mean.tolist(),
        "acc_std": acc_std.tolist(),
    }

    save_gns_npz(os.path.join(args.out_dir, "train.npz"), train_positions, args.particle_type)
    save_gns_npz(os.path.join(args.out_dir, "test.npz"), test_positions, args.particle_type)
    with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"train.npz: {train_positions.shape}")
    print(f"test.npz:  {test_positions.shape}")
    print(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
