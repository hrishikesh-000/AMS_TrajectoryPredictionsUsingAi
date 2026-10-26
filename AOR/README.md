# AOR Pipeline — Prediction of Particle Trajectories using AI

This folder holds the AOR (Angle of Repose) half of the project: converting raw
LIGGGHTS DEM output into GNS training data, training/evaluating the GNS
surrogate honestly (not on a trivially-settled tail), and an Ensemble Kalman
Filter (EnKF) correction layer that fixes long-rollout drift in the surrogate.

It builds directly on [geoelements/gns](https://github.com/geoelements/gns),
the open-source PyTorch Graph Network Simulator (Kumar & Vantassel, 2023),
itself based on Sanchez-Gonzalez et al. (2020), *Learning to Simulate Complex
Physics with Graph Networks*.

## Setup

This is not a standalone package — it's a small set of scripts meant to sit
inside a `geoelements/gns` checkout:

```bash
git clone https://github.com/geoelements/gns.git
cd gns
# copy this folder's contents in:
cp AOR/utils/*.py utils/
cp AOR/visualize_trajectories.py .
```

`utils/enkf_rollout.py` imports directly from `gns.train` and
`gns.data_loader`, so it needs to live at `<gns_repo>/utils/enkf_rollout.py`
for those imports to resolve (same pattern `gns/train.py` itself uses).

Dependencies: whatever `geoelements/gns` already requires (PyTorch,
PyTorch Geometric, NumPy), plus `scipy` (for `shape_and_spacing.py`) and
`matplotlib` (for the visualization/plotting scripts).

## What's here

| Script | Purpose |
|---|---|
| `utils/convert_aor_csv_to_npz.py` | Converts raw per-frame LIGGGHTS CSVs into the `train.npz` / `test.npz` / `metadata.json` format the GNS dataloader expects. Computes velocity/acceleration normalization stats from the train split only. |
| `utils/enkf_rollout.py` | The Ensemble Kalman Filter correction layer. Runs an N-member ensemble of perturbed GNS rollouts, periodically corrects each member toward a noised observation of the true position via a Kalman gain computed from the ensemble's own spread (diagonal/localized covariance — independent per particle, per axis). Reuses `_get_simulator` from `gns/train.py` so the model is built identically to how it was trained. |
| `utils/compute_aor_enkf_rmse.py` | Loads a baseline and EnKF rollout pkl pair (as saved by `enkf_rollout.py`) and computes per-frame and aggregate RMSE/MSE, saving the result to JSON for plotting. This is what produced the numbers in the results table below. |
| `utils/count_particles_per_frame.py` | Row count per CSV — confirms the particle-injection schedule (batches arriving over time). |
| `utils/velocity_profile.py` | Mean/max/percentile particle speed per frame — used to find where the simulation actually settles (i.e. stops being a meaningful dynamics test). |
| `utils/batch_cohort_speed.py` | Per-injection-batch mean speed over time, rather than the whole-system average. |
| `utils/shape_and_spacing.py` | Final-frame particle positions (shape snapshot), grain-size vs. position (segregation check), and nearest-neighbor distance distribution (used to sanity-check the `connectivity_radius` hyperparameter against real particle spacing). |
| `visualize_trajectories.py` | Renders side-by-side ground-truth vs. prediction 3D animated GIFs, colored by displacement. |

## Why "honest" evaluation

The raw 400-frame trajectory settles into a static pile well before frame
400. A train/test split that puts the test set entirely in that static tail
(e.g. training on frames 1–200, testing on 300–400) makes the model look
almost perfect (R² ≈ 0.999) without ever testing it on the genuinely dynamic,
harder part of the simulation — because "predicting" a pile that isn't
moving is trivial.

The splits used for the results below instead train on an early, dynamic
window and evaluate across the full remaining rollout, well past where the
system settles, so the reported numbers reflect real predictive difficulty
rather than an easy static tail.

## Pipeline walkthrough

```bash
# 1. Convert raw CSVs to GNS format
python utils/convert_aor_csv_to_npz.py \
    --csv_dir=all_data_01_AOR --n_particles=3000 \
    --train_start=41 --train_end=160 \
    --test_start=41 --test_end=400 \
    --particle_type=5 --connectivity_radius=0.006 --dt=0.0001 \
    --out_dir=data/aor_3000p

# 2. Train (standard gns.train, unmodified)
python -m gns.train --mode=train \
    --data_path=data/aor_3000p/ --model_path=models/aor_3000p/ \
    --output_path=rollouts/aor_3000p/ --ntraining_steps=200000

# 3a. Baseline rollout, no correction
python utils/enkf_rollout.py \
    --data_dir=data/aor_3000p --model_path=models/aor_3000p/model-200000.pt \
    --out_dir=rollouts/aor_3000p_baseline --n_ensemble=2 --disable_correction

# 3b. EnKF-corrected rollout
python utils/enkf_rollout.py \
    --data_dir=data/aor_3000p --model_path=models/aor_3000p/model-200000.pt \
    --out_dir=rollouts/aor_3000p_enkf --n_ensemble=30 --correction_interval=20

# 4. Visualize
python visualize_trajectories.py rollouts/aor_3000p_enkf/rollout_enkf.pkl out.gif 3000
```

## Results (full-range rollout, correction interval = 20)

| Particles | Rollout length | Baseline RMSE | EnKF RMSE | R² (baseline → EnKF) | Improvement |
|---|---|---|---|---|---|
| 3000 | 354 steps | 5.03e-03 m | 5.40e-04 m | 0.850 → 0.998 | 9.3x |
| 4000 | 344 steps | 5.97e-03 m | 4.62e-04 m | 0.790 → 0.999 | 12.9x |

Full presented results, charts and rollout videos are in the accompanying
presentation (`AMS_Final.pptx`, slides 6–12).

## Data

Raw DEM CSVs (`all_data_01_AOR/`) are not included here — ~800MB of raw
per-frame simulation output. Point `--csv_dir` at wherever your copy lives;
the conversion script above regenerates the `.npz`/`metadata.json` files
deterministically from it.
