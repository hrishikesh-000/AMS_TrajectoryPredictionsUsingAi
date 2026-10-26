"""Compute per-frame particle speed statistics (mean, max, p10/p50/p90) from raw
LIGGGHTS DEM CSV output, for visualizing the dynamic-to-static transition.

Usage:
  python velocity_profile.py --csv_dir=all_data_01_AOR --out=speed_aor.json
"""
import argparse
import csv
import json
import os
import re

import numpy as np


def compute_frame_stats(csv_dir):
    pattern = re.compile(r"_(\d+)\.csv$")
    frame_stats = {}
    for fname in os.listdir(csv_dir):
        m = pattern.search(fname)
        if not m:
            continue
        frame = int(m.group(1))
        path = os.path.join(csv_dir, fname)
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            speeds = [
                (float(r["v:0"]) ** 2 + float(r["v:1"]) ** 2 + float(r["v:2"]) ** 2) ** 0.5
                for r in reader
            ]
        speeds = np.array(speeds)
        frame_stats[frame] = {
            "mean": float(speeds.mean()),
            "max": float(speeds.max()),
            "p10": float(np.percentile(speeds, 10)),
            "p50": float(np.percentile(speeds, 50)),
            "p90": float(np.percentile(speeds, 90)),
        }
    return dict(sorted(frame_stats.items()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    stats = compute_frame_stats(args.csv_dir)
    with open(args.out, "w") as f:
        json.dump(stats, f)
    print(f"Wrote {len(stats)} frames to {args.out}")


if __name__ == "__main__":
    main()
