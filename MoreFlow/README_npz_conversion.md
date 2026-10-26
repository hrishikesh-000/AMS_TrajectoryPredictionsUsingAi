# CSV → NPZ Conversion (3000-particle dynamic experiment)

This document covers **only** the data-conversion step: turning the raw
per-frame DEM simulation output into the `.npz` / `metadata.json` format
the [geoelements/gns](https://github.com/geoelements/gns) repo expects for
training and rollout. It does not cover training, rollout, or rendering.

## Source data

```
C:\Users\Jayakrishna A J\GNS_Project\all_data_02_more_Flow\
```

400 CSV files, one per simulation frame, named:

```
particles_20260218_1.csv
particles_20260218_2.csv
...
particles_20260218_400.csv
```

The trailing number is the frame index. Each row in a CSV is one particle
at that frame, with 16 columns:

```
id, type, mass, v:0, v:1, v:2, f:0, f:1, f:2, radius,
omega:0, omega:1, omega:2, Points:0, Points:1, Points:2
```

Only `Points:0 / Points:1 / Points:2` (x, y, z position) are used. Velocity,
force, mass, radius, and angular velocity are ignored — the GNS dataloader
only needs raw positions and derives velocity/acceleration itself by
differencing consecutive frames.

Frames don't all have the same particle count (particles are added over
time as the simulation runs), so a fixed particle count (e.g. 3000) is only
valid from the frame where that many particles first exist.

## Script

```
C:\Users\Jayakrishna A J\GNS_Project\build_all_dynamic.py
```

Run it with the `gns_env` conda environment (needs `numpy`, `pandas`):

```bat
conda activate gns_env
cd /d "C:\Users\Jayakrishna A J\GNS_Project"
python build_all_dynamic.py
```

### What it does

1. Globs all CSVs in `all_data_02_more_Flow\`, sorted numerically by frame
   number.
2. For each frame in a given range, reads the CSV, keeps only the first
   `N_PARTICLES` rows (3000) and only the `Points:0/1/2` columns, casts to
   `float32`.
3. Stacks the selected frames into one array of shape
   `(num_frames, 3000, 3)` — `(time, particles, xyz)`.
4. Wraps each split as `simulation_0 = [positions_array, particle_type_array]`
   and writes it with `np.savez_compressed(...)`.
5. Computes `metadata.json` from the combined data (bounds, velocity
   mean/std from frame-to-frame differencing, fixed `dt` and connectivity
   radius).

### Frame split (one continuous trajectory, frames 41–400)

| Frames | Frame count | File | Purpose |
|---|---|---|---|
| 41–200 | 160 | `train.npz` | active flow / settling transient |
| 201–300 | 100 | `valid.npz` | settled ring, early stopping |
| 301–400 | 100 | `test.npz` | settled ring — "static" evaluation (used to pick the connectivity radius: 1x–5x sweep, 5x/0.0060 won on RMSE/MAE/R²) |

### Two output datasets

**`dataset_3000_5x\`** — the base dataset, used to train the model. Contains
the full train/valid/test split above.

**`dataset_dynamic_3000\`** — same `train.npz`/`valid.npz` (byte-identical
copies, so it's the same trained checkpoint), but `test.npz` is replaced
with the 41–200 window run as **one continuous 160-frame sequence** instead
of individual training pairs. This is the actual "dynamic experiment" test
set: it asks the trained model to reproduce the funnel → ring collapse from
scratch via autoregressive rollout, rather than just holding an
already-settled ring in place (which is all the static test at 301–400
checks for).

## Output format

Each `.npz` contains one key, `simulation_0`, holding a 2-element object
array:

```python
[positions, particle_type]
# positions:     float32, shape (num_frames, 3000, 3)
# particle_type: int,     shape (3000,)  -- all zeros (single material)
```

`metadata.json` fields:

| Field | Value | Meaning |
|---|---|---|
| `bounds` | per-axis `[min, max]` | computed over the full 41–400 trajectory (train+valid+test), so normalization covers both the flowing and settled states |
| `sequence_length` | 160 | length of `train.npz` (and of `dataset_dynamic_3000`'s `test.npz`) |
| `default_connectivity_radius` | 0.0060 | 5x — best RMSE/MAE/R² from the radius sweep |
| `dim` | 3 | 3D simulation |
| `dt` | 0.0025 | fixed simulation timestep |
| `vel_mean` / `vel_std` | per-axis | computed by differencing consecutive frames across the full concatenated trajectory (valid across split boundaries since frames 200→201 and 300→301 are contiguous in the real simulation) |
| `acc_mean` / `acc_std` | `[0,0,0]` / `[1,1,1]` | GNS convention — acceleration left unnormalized at the metadata level |

## Verifying a rebuild

Re-running `build_all_dynamic.py` is deterministic and idempotent — it
reads the same source CSVs and will reproduce byte-identical `.npz`/
`metadata.json` output. Useful sanity check after any edit to the script:

```python
import numpy as np, json
z = np.load("dataset_dynamic_3000/test.npz", allow_pickle=True)
pos = z["simulation_0"][0]
print(pos.shape)  # expect (160, 3000, 3)

meta = json.load(open("dataset_dynamic_3000/metadata.json"))
print(meta["sequence_length"], meta["default_connectivity_radius"])  # 160 0.006
```
