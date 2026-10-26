"""Count number of particles (rows minus header) in each frame CSV for one or more
directories of LIGGGHTS-style particle CSVs, and dump the result as JSON for plotting.

Usage:
  python count_particles_per_frame.py --csv_dir=all_data_01_AOR --out=counts_aor.json
"""
import argparse
import json
import os
import re


def count_particles(csv_dir):
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
        frame_counts[frame] = n_lines - 1  # minus header
    return dict(sorted(frame_counts.items()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    counts = count_particles(args.csv_dir)
    with open(args.out, "w") as f:
        json.dump(counts, f)
    print(f"Wrote {len(counts)} frames to {args.out}")


if __name__ == "__main__":
    main()
