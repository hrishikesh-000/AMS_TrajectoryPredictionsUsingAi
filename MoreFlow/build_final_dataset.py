"""
build_final_dataset.py

The "foolproof" replacement for every earlier NPZ builder in this project.
Fixes, relative to build_dynamic_split.py / build_extended_test.py:

1. PARTICLE IDENTITY IS MATCHED BY THE `id` COLUMN, NOT ROW POSITION.
   Earlier scripts did `df.iloc[:N_PARTICLES]`, silently assuming row i is
   always the same physical particle across every frame. This script
   verifies that assumption instead of assuming it: it checks, for every
   single frame in range, that the `id` column is exactly the contiguous
   set {1, ..., N_PARTICLES} (no gaps, no duplicates, no reordering), and
   explicitly filters+sorts by `id` before taking positions. If a frame
   ever violates this, the script raises instead of silently producing
   wrong data.

   (Verified once, empirically, across all 400 CSVs before writing this
   script: id is always exactly `1..n_particles_in_frame` ascending, and
   particle count never decreases frame-to-frame -- i.e. particles are
   only ever appended, never removed/reindexed. So "first N rows" and
   "id in 1..N" happen to coincide today, but this script no longer
   *relies* on that coincidence.)

2. TRAIN NOW COVERS BOTH PHYSICAL REGIMES.
   The previous checkpoint trained only on frames 41-160 (pure active
   inflow/collision), then was asked to roll out through 161-400, which
   includes the settled/static ring -- a regime it had zero training
   examples of. That distribution shift, stacked on top of a rollout 2x
   longer than anything trained on, is the direct cause of the
   diverging/"flying out of frame" particles seen in that gif.

   New split pulls both regimes into train:
       train: frames  41-300 (260 frames) -- active flow AND
                                              transition AND settled ring
       test : frames 301-400 (100 frames) -- held-out continuation,
                                              entirely inside a regime
                                              the model has already seen
                                              examples of.
   This matches the very first split design used in this project
   (dataset_3000_5x: train 41-200 / valid 201-300 / test 301-400) with
   valid folded into train, per request for train+test only.

3. NO GAP-FRAME OR MISSING-FILE BLIND SPOTS.
   Every integer frame number in [TRAIN_START, TEST_END] must have a
   corresponding CSV, or the script raises immediately naming the first
   missing frame, instead of silently skipping it (the old `continue` on
   short frames could hide a missing frame the same way).

4. NO NaN/Inf, NO LEAKAGE INTO NORMALIZATION STATS.
   Positions are checked for finiteness. velocity mean/std (used to
   normalize the network's targets) are computed from TRAIN ONLY, not
   train+test combined as before -- using test statistics to normalize
   training targets is a (small) form of leakage.
   `bounds` (the physical container extent used for the boundary/graph
   radius) is still taken from the full 41-400 range, since it describes
   a fixed physical apparatus, not a statistic of the modeled quantity.
"""

import glob
import json
import os
import re

import numpy as np
import pandas as pd

CSV_FOLDER = r"C:\Users\Jayakrishna A J\GNS_Project\all_data_02_more_Flow"
OUT_DIR = r"C:\Users\Jayakrishna A J\GNS_Project\FINAL_OUTPUT\dataset"
POSITION_COLS = ["Points:0", "Points:1", "Points:2"]

N_PARTICLES = 3000
TRAIN_START, TRAIN_END = 41, 300
TEST_START, TEST_END = 301, 400

RADIUS = 0.0060  # 5x -- carried over from the validated 1x-5x radius sweep
DT = 0.0025


def index_files(folder):
    files = glob.glob(os.path.join(folder, "*.csv"))
    by_frame = {}
    for f in files:
        m = re.findall(r"_(\d+)\.csv", f)
        if not m:
            continue
        by_frame[int(m[-1])] = f
    return by_frame


def load_frame(by_frame, frame_no):
    if frame_no not in by_frame:
        raise SystemExit(
            f"FATAL: missing CSV for frame {frame_no} "
            f"(expected particles_*_{frame_no}.csv in {CSV_FOLDER})"
        )
    df = pd.read_csv(by_frame[frame_no])
    if "id" not in df.columns:
        raise SystemExit(f"FATAL: frame {frame_no} CSV has no 'id' column")

    df = df[df["id"].between(1, N_PARTICLES)].sort_values("id")

    ids = df["id"].to_numpy()
    expected_ids = np.arange(1, N_PARTICLES + 1)
    if ids.shape[0] != N_PARTICLES or not np.array_equal(ids, expected_ids):
        raise SystemExit(
            f"FATAL: frame {frame_no} does not contain a clean, contiguous "
            f"set of ids 1..{N_PARTICLES}. Found {ids.shape[0]} matching "
            f"rows. Refusing to silently guess -- fix the source data or "
            f"lower N_PARTICLES."
        )

    pos = df[POSITION_COLS].to_numpy(dtype=np.float32)
    if not np.all(np.isfinite(pos)):
        raise SystemExit(f"FATAL: frame {frame_no} has NaN/Inf in positions")

    return pos


def load_range(by_frame, lo, hi, label):
    print(f"Loading {label} frames {lo}-{hi} (id-matched, {N_PARTICLES} particles each) ...")
    frames = [load_frame(by_frame, fn) for fn in range(lo, hi + 1)]
    arr = np.stack(frames, axis=0)
    print(f"  {label}: {arr.shape} (expect ({hi - lo + 1}, {N_PARTICLES}, 3))")
    return arr


def save_npz(pos, path):
    sim = np.array([pos, np.zeros(N_PARTICLES, dtype=int)], dtype=object)
    np.savez_compressed(path, simulation_0=sim)
    print(f"  saved -> {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    by_frame = index_files(CSV_FOLDER)
    if not by_frame:
        raise SystemExit(f"No CSVs found in {CSV_FOLDER}")

    train = load_range(by_frame, TRAIN_START, TRAIN_END, "train")
    test = load_range(by_frame, TEST_START, TEST_END, "test")

    save_npz(train, os.path.join(OUT_DIR, "train.npz"))
    save_npz(test, os.path.join(OUT_DIR, "test.npz"))

    train_vel = np.diff(train, axis=0)
    all_pos = np.concatenate([train, test], axis=0)  # bounds only: fixed physical apparatus

    meta = {
        "bounds": [
            [float(all_pos[:, :, i].min()), float(all_pos[:, :, i].max())]
            for i in range(3)
        ],
        "sequence_length": None,  # auto-detect from whichever split GNS reads
        "default_connectivity_radius": RADIUS,
        "dim": 3,
        "dt": DT,
        "vel_mean": train_vel.mean(axis=(0, 1)).tolist(),
        "vel_std": train_vel.std(axis=(0, 1)).tolist(),
        "acc_mean": [0.0, 0.0, 0.0],
        "acc_std": [1.0, 1.0, 1.0],
        "train_frame_range": [TRAIN_START, TRAIN_END],
        "test_frame_range": [TEST_START, TEST_END],
        "n_particles": N_PARTICLES,
    }
    with open(os.path.join(OUT_DIR, "metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Saved metadata -> {os.path.join(OUT_DIR, 'metadata.json')}")
    print("Done.")


if __name__ == "__main__":
    main()
