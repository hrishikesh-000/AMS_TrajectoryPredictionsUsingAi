import json
import os

output_folder = r"C:\Users\Jayakrishna A J\GNS_Project\dataset_1000_v3"

# Overall boundary computed from all 400 frames
bounds = [
    [-0.049303, 0.049301],
    [-0.049303, 0.049304],
    [-0.079312, 0.010198]
]

# Base metadata values (reuse vel_mean/std from dataset_1000_v2 if available,
# otherwise recompute — see note below)
base_metadata = {
    "bounds": bounds,
    "sequence_length": 190,   # train frames: 11-200
    "dim": 3,
    "dt": 0.0025,
    "vel_mean": [-6.13e-07, 1.75e-07, -3.53e-05],
    "vel_std":  [8.31e-05, 8.19e-05, 3.20e-04],
    "acc_mean": [0.0, 0.0, 0.0],
    "acc_std":  [1.0, 1.0, 1.0]
}

radius_multiples = {
    "1x": 0.0012,
    "2x": 0.0024,
    "3x": 0.0036,
    "4x": 0.0048,
    "5x": 0.0060
}

for label, radius in radius_multiples.items():
    metadata = dict(base_metadata)
    metadata["default_connectivity_radius"] = radius
    
    filename = f"metadata_{label}.json"
    filepath = os.path.join(output_folder, filename)
    
    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Saved {filename} with radius = {radius}")

print("\nAll 5 metadata files created in:", output_folder)