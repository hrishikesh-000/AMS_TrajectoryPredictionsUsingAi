"""Compute per-frame and aggregate RMSE/MSE for the AOR 3000p/4000p full-range EnKF
(interval=20) runs, from the raw rollout pkls, and save to JSON for plotting.

Expects, under --tmp_dir (default C:/tmp):
  rollouts_aor_3000p_fullrange/rollout_ex0_clean_matched.pkl          (baseline, 3000p)
  rollouts_aor_3000p_fullrange_enkf_int20/rollout_enkf_clean.pkl      (EnKF, 3000p)
  rollouts_aor_4000p_split1_fullrange/rollout_ex0_clean_matched.pkl   (baseline, 4000p)
  rollouts_aor_4000p_split1_fullrange_enkf/rollout_enkf_clean.pkl     (EnKF, 4000p)
Each pkl is a dict with numpy arrays "predicted_rollout" and "ground_truth_rollout",
shape (n_frames, n_particles, 3).

Usage:
  python compute_aor_enkf_rmse.py --out=aor_rmse_data.json
"""
import argparse
import json
import os
import pickle

import numpy as np

CONFIGS = {
    "3000p": {
        "baseline_dir": "rollouts_aor_3000p_fullrange",
        "baseline_file": "rollout_ex0_clean_matched.pkl",
        "enkf_dir": "rollouts_aor_3000p_fullrange_enkf_int20",
        "enkf_file": "rollout_enkf_clean.pkl",
    },
    "4000p": {
        "baseline_dir": "rollouts_aor_4000p_split1_fullrange",
        "baseline_file": "rollout_ex0_clean_matched.pkl",
        "enkf_dir": "rollouts_aor_4000p_split1_fullrange_enkf",
        "enkf_file": "rollout_enkf_clean.pkl",
    },
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tmp_dir", default=r"C:\tmp")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    out = {}
    for label, cfg in CONFIGS.items():
        with open(os.path.join(args.tmp_dir, cfg["baseline_dir"], cfg["baseline_file"]), "rb") as f:
            base_d = pickle.load(f)
        with open(os.path.join(args.tmp_dir, cfg["enkf_dir"], cfg["enkf_file"]), "rb") as f:
            enkf_d = pickle.load(f)

        base_pred, base_gt = base_d["predicted_rollout"], base_d["ground_truth_rollout"]
        enkf_pred, enkf_gt = enkf_d["predicted_rollout"], enkf_d["ground_truth_rollout"]

        base_rmse_agg = float(np.sqrt(np.mean((base_pred - base_gt) ** 2)))
        enkf_rmse_agg = float(np.sqrt(np.mean((enkf_pred - enkf_gt) ** 2)))
        base_mse_agg = float(np.mean((base_pred - base_gt) ** 2))
        enkf_mse_agg = float(np.mean((enkf_pred - enkf_gt) ** 2))

        base_rmse_frame = np.sqrt(np.mean((base_pred - base_gt) ** 2, axis=(1, 2)))
        enkf_rmse_frame = np.sqrt(np.mean((enkf_pred - enkf_gt) ** 2, axis=(1, 2)))

        print(f"{label}: baseline RMSE={base_rmse_agg:.6e}  enkf RMSE={enkf_rmse_agg:.6e}  "
              f"improvement={base_rmse_agg/enkf_rmse_agg:.2f}x  n_frames={len(base_rmse_frame)}")

        out[label] = {
            "base_rmse_frame": base_rmse_frame.tolist(),
            "enkf_rmse_frame": enkf_rmse_frame.tolist(),
            "base_rmse_agg": base_rmse_agg,
            "enkf_rmse_agg": enkf_rmse_agg,
            "base_mse_agg": base_mse_agg,
            "enkf_mse_agg": enkf_mse_agg,
        }

    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
