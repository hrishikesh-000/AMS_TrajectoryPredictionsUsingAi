"""For each injection batch (a contiguous ID range), compute that cohort's own mean
particle speed at every frame from its injection onward. Lets you compare how
different batches (injected at different times) individually decay toward settling,
rather than only looking at the whole-system aggregate.

Usage:
  python batch_cohort_speed.py --csv_dir=all_data_01_AOR --out=cohort_aor.json
"""
import argparse
import csv
import json
import os
import re

import numpy as np


def batch_boundaries(csv_dir):
    """Return sorted frames and cumulative particle count per frame, plus the
    (batch_index -> (start_id, end_id, inject_frame)) boundaries."""
    pattern = re.compile(r"_(\d+)\.csv$")
    frame_counts = {}
    for fname in os.listdir(csv_dir):
        m = pattern.search(fname)
        if not m:
            continue
        frame = int(m.group(1))
        path = os.path.join(csv_dir, fname)
        with open(path, "rb") as f:
            n_lines = sum(1 for _ in f)
        frame_counts[frame] = n_lines - 1
    frames = sorted(frame_counts)

    batches = []
    prev_total = 0
    for f in frames:
        c = frame_counts[f]
        if c != prev_total:
            batches.append({"inject_frame": f, "start_id": prev_total + 1, "end_id": c})
            prev_total = c
    return frames, batches


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    frames, batches = batch_boundaries(args.csv_dir)
    upper_bounds = np.array([b["end_id"] for b in batches])

    n_batches = len(batches)
    # per batch: {frame: mean_speed}
    cohort_series = [dict() for _ in range(n_batches)]

    pattern = re.compile(r"_(\d+)\.csv$")
    for fname in os.listdir(args.csv_dir):
        m = pattern.search(fname)
        if not m:
            continue
        frame = int(m.group(1))
        path = os.path.join(args.csv_dir, fname)
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            ids = []
            speeds = []
            for r in reader:
                ids.append(int(r["id"]))
                v0, v1, v2 = float(r["v:0"]), float(r["v:1"]), float(r["v:2"])
                speeds.append((v0 * v0 + v1 * v1 + v2 * v2) ** 0.5)
        ids = np.array(ids)
        speeds = np.array(speeds)
        batch_idx = np.searchsorted(upper_bounds, ids, side="left")

        for b in range(n_batches):
            mask = batch_idx == b
            if mask.any():
                cohort_series[b][frame] = float(speeds[mask].mean())

    out = {
        "batches": [
            {
                "n": i + 1,
                "inject_frame": batches[i]["inject_frame"],
                "start_id": batches[i]["start_id"],
                "end_id": batches[i]["end_id"],
                "series": dict(sorted(cohort_series[i].items())),
            }
            for i in range(n_batches)
        ]
    }
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"Wrote {n_batches} batch cohorts to {args.out}")


if __name__ == "__main__":
    main()
