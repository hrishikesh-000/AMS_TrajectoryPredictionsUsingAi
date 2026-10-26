import numpy as np, pandas as pd, glob, os, re, json

csv_folder = r"C:\Users\Jayakrishna A J\GNS_Project\all_data_02_more_Flow"
POSITION_COLS = ['Points:0', 'Points:1', 'Points:2']
experiments = {1000: 11, 2000: 21, 3000: 41, 4000: 51}
VALID_START, VALID_END, TEST_START, TEST_END = 201, 300, 301, 400
RADIUS = 0.0060  # 5x — using literature-consistent radius

files = glob.glob(os.path.join(csv_folder, "*.csv"))
files = sorted(files, key=lambda f: int(re.findall(r"_(\d+)\.csv", f)[-1]))

for n, start_frame in experiments.items():
    out = rf"C:\Users\Jayakrishna A J\GNS_Project\dataset_{n}_5x"
    os.makedirs(out, exist_ok=True)
    print(f"\n=== {n} particles (start frame {start_frame}, radius={RADIUS}) ===")
    train, valid, test = [], [], []
    for f in files:
        fn = int(re.findall(r"_(\d+)\.csv", f)[-1])
        df = pd.read_csv(f)
        if len(df) < n:
            continue
        pos = df.iloc[:n][POSITION_COLS].values.astype(np.float32)
        if start_frame <= fn <= 200:
            train.append(pos)
        elif VALID_START <= fn <= VALID_END:
            valid.append(pos)
        elif TEST_START <= fn <= TEST_END:
            test.append(pos)
    train, valid, test = np.array(train), np.array(valid), np.array(test)
    print(f"Train:{len(train)} Valid:{len(valid)} Test:{len(test)}")

    def save(p, path):
        sim = np.array([p, np.zeros(n, dtype=int)], dtype=object)
        np.savez_compressed(path, simulation_0=sim)

    save(train, os.path.join(out, 'train.npz'))
    save(valid, os.path.join(out, 'valid.npz'))
    save(test, os.path.join(out, 'test.npz'))

    allp = np.concatenate([train, valid, test], axis=0)
    vels = np.diff(allp, axis=0)
    meta = {
        "bounds": [[float(allp[:, :, i].min()), float(allp[:, :, i].max())] for i in range(3)],
        "sequence_length": len(train),
        "default_connectivity_radius": RADIUS,
        "dim": 3, "dt": 0.0025,
        "vel_mean": vels.mean(axis=(0, 1)).tolist(),
        "vel_std": vels.std(axis=(0, 1)).tolist(),
        "acc_mean": [0.0, 0.0, 0.0], "acc_std": [1.0, 1.0, 1.0]
    }
    json.dump(meta, open(os.path.join(out, 'metadata.json'), 'w'), indent=2)
    print(f"✅ Saved {n} | bounds X:[{meta['bounds'][0][0]:.4f},{meta['bounds'][0][1]:.4f}]")

print("\n✅ ALL 4 DATASETS BUILT (radius=0.0060)")