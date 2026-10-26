"""Compute three things from raw LIGGGHTS DEM CSVs, all cheap, all needle-moving:
1. Final-frame particle positions (shape snapshot: dome vs donut).
2. Grain-size (radius) vs spatial position, to check for segregation.
3. Nearest-neighbor distance distribution at a few frames, to sanity-check the
   connectivity_radius hyperparameter against real particle spacing.

Usage:
  python shape_and_spacing.py --csv_dir=all_data_01_AOR --out=shape_aor.json
"""
import argparse
import csv
import json

from scipy.spatial import cKDTree
import numpy as np


def read_frame(csv_dir, frame):
    path = f"{csv_dir}/particles_20260218_{frame}.csv"
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        ids, radii, pts = [], [], []
        for r in reader:
            ids.append(int(r["id"]))
            radii.append(float(r["radius"]))
            pts.append([float(r["Points:0"]), float(r["Points:1"]), float(r["Points:2"])])
    return np.array(ids), np.array(radii), np.array(pts)


def nn_distances(pts):
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2)
    return dists[:, 1]  # skip self (distance 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_dir", required=True)
    p.add_argument("--shape_frame", type=int, default=400)
    p.add_argument("--nn_frames", type=int, nargs="+", default=[10, 100, 400])
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ids, radii, pts = read_frame(args.csv_dir, args.shape_frame)

    # 1. Shape snapshot
    shape = {
        "frame": args.shape_frame,
        "x": pts[:, 0].round(6).tolist(),
        "y": pts[:, 1].round(6).tolist(),
        "z": pts[:, 2].round(6).tolist(),
    }

    # 2. Grain-size segregation (reuse same frame)
    radius_levels = sorted(set(round(r, 6) for r in radii))
    segregation = {"frame": args.shape_frame, "radius_levels": radius_levels, "groups": []}
    for rl in radius_levels:
        mask = np.isclose(radii, rl)
        gx, gy, gz = pts[mask, 0], pts[mask, 1], pts[mask, 2]
        segregation["groups"].append({
            "radius": rl,
            "n": int(mask.sum()),
            "x": gx.round(6).tolist(),
            "y": gy.round(6).tolist(),
            "z": gz.round(6).tolist(),
            "mean_z": float(gz.mean()),
            "std_z": float(gz.std()),
            "mean_r_xy": float(np.sqrt(gx**2 + gy**2).mean()),
        })

    # 3. Nearest-neighbor distances at a few frames
    nn = {"frames": {}}
    for f in args.nn_frames:
        _, _, fpts = read_frame(args.csv_dir, f)
        d = nn_distances(fpts)
        hist, edges = np.histogram(d, bins=60, range=(0, 0.01))
        nn["frames"][f] = {
            "hist": hist.tolist(),
            "edges": edges.round(6).tolist(),
            "mean": float(d.mean()),
            "p50": float(np.percentile(d, 50)),
            "p90": float(np.percentile(d, 90)),
        }

    out = {"shape": shape, "segregation": segregation, "nn": nn}
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
