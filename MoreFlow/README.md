# GNS Surrogate Model + EnKF Correction — DEM Particle Simulation

End-to-end pipeline that trains a Graph Network Simulator (GNS,
[geoelements/gns](https://github.com/geoelements/gns)) as a surrogate for
DEM particle simulations, then wraps the trained model in an Ensemble
Kalman Filter (EnKF) to correct rollout drift, and produces a full
before/after comparison (metrics table, plot, and side-by-side GIF).

The pipeline is dataset-agnostic in principle but is currently exercised
at three particle counts — **3,000 / 4,000 / 5,000 particles** — from
separate DEM simulation runs. This README documents the whole pipeline
and, in particular, exactly what to change to run it for each of the
three particle counts.

---

## 1. Repository layout

```
FINAL_OUTPUT/
├── dataset/                    # produced by build_final_dataset.py (train.npz, test.npz, metadata.json)
├── models/                     # GNS checkpoints (model-*.pt), written by gns.train
├── train_mode_rollout_scratch/ # unused output_path required by gns.train --mode=train
├── checkpoint_eval/            # eval_checkpoints.py output + best_checkpoint.json
├── results/
│   ├── enkf_rollout/           # ground_truth.npy, plain_trajectories.npy, enkf_trajectories.npy
│   ├── gifs/                   # enkf_comparison.gif
│   ├── training_time_seconds.txt
│   ├── final_results.json / .txt
│   ├── drift_comparison_table.csv / .txt
│   └── drift_comparison_plot.png
├── logs/                       # SLURM stdout/stderr
└── scripts/
    ├── build_final_dataset.py      # CSV -> train.npz / test.npz / metadata.json
    ├── build_5_metadata.py         # legacy: connectivity-radius sweep metadata (1x–5x) for a 1,000-particle dataset
    ├── gns_enkf_rollout.py         # plain GNS rollout AND EnKF-corrected rollout
    ├── drift_comparison_report.py  # RMSE/MSE/R2 table + plots, before vs after
    ├── make_comparison_gif.py      # 3-panel Reality | Before | After GIF
    ├── eval_checkpoints.py         # (referenced by run_pipeline.slurm — not included here)
    ├── metrics_utils.py            # (imported by drift_comparison_report.py — not included here)
    └── final_report.py             # (referenced by run_pipeline.slurm — not included here)
run_pipeline.slurm                  # sbatch entry point tying stages 1–6 together
```

> `eval_checkpoints.py`, `metrics_utils.py`, and `final_report.py` are
> referenced by `run_pipeline.slurm` / `drift_comparison_report.py` but
> were not part of the files reviewed for this README — make sure they
> exist in `scripts/` before submitting the SLURM job.

---

## 2. Pipeline stages (what `run_pipeline.slurm` does)

1. **Train** the GNS from scratch (or resume) on `dataset/train.npz` via
   `gns.train --mode=train`. Requeue-safe: resubmitting the same job
   resumes from the latest checkpoint.
2. **Checkpoint sweep**: `eval_checkpoints.py` rolls out every saved
   checkpoint against `test.npz` and picks the one with the best rollout
   RMSE (this stands in for early stopping — no `valid.npz` is used).
3. **Plain rollout** ("before"): `gns_enkf_rollout.py --mode=plain` with
   the selected checkpoint.
4. **EnKF-corrected rollout** ("after"): `gns_enkf_rollout.py --mode=enkf`,
   same checkpoint, periodic ensemble Kalman correction against sparsely
   observed ground truth.
5. **Final metrics report**: `final_report.py` — training time + best
   checkpoint + before/after summary.
6. **Comparison GIF**: `make_comparison_gif.py` — 3-panel
   Reality / Before / After animation.

`drift_comparison_report.py` is a standalone add-on you can run any time
after stage 4 to get the full per-frame RMSE table (`.csv`/`.txt`) and the
3-panel drift plot (`.png`); it isn't currently wired into the SLURM
script.

---

## 3. Dataset preparation — `build_final_dataset.py`

Reads per-frame CSVs (`Points:0/1/2` position columns + an `id` column),
verifies `id` is exactly the contiguous set `{1..N_PARTICLES}` in every
frame (no silent row-position assumptions), and writes:

- `train.npz` — frames `TRAIN_START–TRAIN_END`
- `test.npz` — frames `TEST_START–TEST_END`
- `metadata.json` — bounds, connectivity radius, `dt`, velocity
  mean/std (computed from **train only**, to avoid leakage), particle
  count, frame ranges.

Key constants at the top of the script you'll edit per run:

| Constant | Meaning |
|---|---|
| `CSV_FOLDER` | Folder of source per-frame CSVs for this simulation |
| `OUT_DIR` | Where `train.npz`/`test.npz`/`metadata.json` are written |
| `N_PARTICLES` | Expected particle count (id range `1..N_PARTICLES`) |
| `TRAIN_START/END`, `TEST_START/END` | Frame split (currently 41–300 train, 301–400 test) |
| `RADIUS` | Connectivity radius used to build the GNN graph |
| `DT` | Simulation timestep |

The script raises immediately (rather than silently producing wrong
data) on: a missing frame, a non-contiguous/duplicate `id` set, or
non-finite positions.

---

## 4. Running for 3,000 / 4,000 / 5,000 particles

The scripts as written are configured for the **3,000-particle**
dataset. To reproduce the pipeline for 4k and 5k, treat each particle
count as an independent dataset + training run with its own paths —
don't overwrite the 3k dataset/checkpoints.

### 4.1 Build the dataset for each particle count

For each `N` in `{3000, 4000, 5000}`, edit `build_final_dataset.py`:

```python
CSV_FOLDER = r"...\all_data_02_more_Flow"   # point at that N's source CSV folder
OUT_DIR    = r"...\FINAL_OUTPUT\dataset_{N}"  # e.g. dataset_3k / dataset_4k / dataset_5k
N_PARTICLES = N                              # 3000 / 4000 / 5000
```

Then run:

```bash
python build_final_dataset.py
```

Confirm the console output shows `(260, N, 3)` for train and
`(100, N, 3)` for test, and that `metadata.json`'s `"n_particles"` field
matches `N`.

> **Connectivity radius:** `RADIUS = 0.0060` is carried over from a
> validated radius sweep at the 3k/1x–5x configuration
> (see `build_5_metadata.py`, which generated the 1x–5x radius
> candidates for that sweep). A denser packing at 4k/5k may warrant
> re-checking whether 0.0060 is still the right radius for that
> particle density rather than assuming it transfers unchanged.

### 4.2 Train + evaluate + rollout for each particle count

Keep each particle count's dataset, checkpoints, and results in their
own directory tree so runs don't collide. The simplest way is to point
`ROOT` (and therefore every downstream path in `run_pipeline.slurm`) at
a per-count subtree:

```bash
# 3,000 particles
ROOT=$HOME/GNS_Project/FINAL_OUTPUT_3k sbatch run_pipeline.slurm

# 4,000 particles
ROOT=$HOME/GNS_Project/FINAL_OUTPUT_4k sbatch run_pipeline.slurm

# 5,000 particles
ROOT=$HOME/GNS_Project/FINAL_OUTPUT_5k sbatch run_pipeline.slurm
```

Each `$ROOT/dataset/` must contain the `train.npz`/`test.npz`/
`metadata.json` produced in step 4.1 for that particle count (either
build directly into `$ROOT/dataset/`, or copy/symlink
`dataset_{N}/` there before submitting).

This keeps `models/`, `checkpoint_eval/`, `results/`, and `logs/`
separate per particle count, so the three runs (and their final
before/after EnKF comparisons) can be compared side by side without
overwriting each other.

### 4.3 Notes specific to scaling particle count

- `gns_enkf_rollout.py` reads `n_particles` from the data itself
  (`ground_truth.shape[1]`), not from a hardcoded constant — no code
  change needed there for 4k/5k.
- EnKF cost scales with particle count: the state dimension is
  `3 * N_PARTICLES`, so the `enkf_update` covariance math
  (`A @ A.T`-shaped terms) gets more expensive at 4k/5k. If a run is
  slow, consider raising `--assim_every` or lowering `--obs_fraction`
  first before assuming the environment is misconfigured.
- Training compute/time also scales with graph size (more nodes/edges
  per message-passing step at higher particle counts) — the SLURM
  `--time=23:59:00` budget that works at 3k may need to be raised for
  4k/5k, and the resume-safe checkpointing (`NSAVE_STEPS`) matters more
  there.

---

## 5. Manual step-by-step (without SLURM)

```bash
# 1. Build the dataset (see §4.1 for per-particle-count edits)
python build_final_dataset.py

# 2. Train (from inside the geoelements/gns checkout)
cd $GNS_REPO_ROOT
python -m gns.train --mode=train \
    --data_path=$DATA_PATH --model_path=$MODEL_PATH \
    --output_path=$TRAIN_ROLLOUT_OUT \
    --ntraining_steps=1000000 --nsave_steps=25000 --batch_size=2 \
    --lr_decay_steps=1000000

# 3. Pick best checkpoint (script not included here — see eval_checkpoints.py)

# 4. Plain rollout ("before")
python gns_enkf_rollout.py \
    --data_path=$DATA_PATH --model_path=$MODEL_PATH \
    --model_file=<best_checkpoint>.pt \
    --output_path=$ENKF_RESULTS_DIR --mode=plain

# 5. EnKF rollout ("after")
python gns_enkf_rollout.py \
    --data_path=$DATA_PATH --model_path=$MODEL_PATH \
    --model_file=<best_checkpoint>.pt \
    --output_path=$ENKF_RESULTS_DIR --mode=enkf \
    --n_ensemble=30 --assim_every=10 --obs_fraction=0.1 \
    --obs_noise_std=1e-4 --init_perturb_std=1e-4

# 6. Drift comparison table + plot
python drift_comparison_report.py --results_dir=$ENKF_RESULTS_DIR

# 7. Comparison GIF
python make_comparison_gif.py --results_dir=$ENKF_RESULTS_DIR
```

---

## 6. Environment

- **Cluster:** ANTS (`ants.cs.ovgu.de`), partition `gpu-stud`, 1×H100
  requested via `--gpus=h100:1`.
- **Toolchain:** Spack-provided CUDA (`source /opt/spack/main/env.sh;
  spack load /wun5tnb`) sourced *before* activating the Python venv.
- **Python env:** plain venv at `$GNS_VENV_PATH` (default `~/gns_env`),
  not conda.
- **GNS repo:** a separate checkout of
  [geoelements/gns](https://github.com/geoelements/gns) at
  `$GNS_REPO_ROOT` (default `~/gns`) — install its own dependencies
  there per its own `requirements.txt`/`setup.py`; the `requirements.txt`
  in this repo only covers the standalone scripts (dataset building,
  EnKF wrapper, reporting, GIF rendering).
- **NCCL:** `torch.distributed` is initialized even for single-GPU
  training; `NCCL_SOCKET_IFNAME=lo`, `NCCL_IB_DISABLE=1`,
  `NCCL_P2P_DISABLE=1` avoid known hangs on multi-NIC HPC nodes.

---

## 7. Known assumptions / things to double check

- `metadata.json`'s `"sequence_length": null` relies on whichever split
  GNS reads to auto-detect its own sequence length.
- `build_5_metadata.py` is a **legacy** helper for a different,
  1,000-particle radius sweep (`dataset_1000_v3`, radii 1x–5x); it is
  unrelated to the 3k/4k/5k particle-count scaling described in §4 and
  is kept here for reference only.
- `nnode_in`/`nedge_in` in `gns_enkf_rollout.py` fall back to fixed
  defaults (`37`/`4` for 3D) when not present in `metadata.json` — this
  depends only on feature dimensionality, not particle count, so no
  change is needed for 4k/5k on that front.
