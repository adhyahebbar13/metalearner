"""Build Datasets/Anomaly/*.csv (features + is_anomaly 0/1) from ADBench npz files and the
repo's old Datasets/Anamoly classification files.

usage: python scripts/make_anomaly_datasets.py <npz_dir> .
<npz_dir> holds ADBench "Classical" .npz files: github.com/Minqi824/ADBench/tree/main/adbench/datasets/Classical
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

npz_dir, repo = Path(sys.argv[1]), Path(sys.argv[2])
out = repo / "Datasets" / "Anomaly"
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
MAX_ROWS = 10000

for f in sorted(npz_dir.glob("*.npz")):
    d = np.load(f, allow_pickle=False)
    X, y = d["X"], d["y"].astype(int)
    if len(X) > MAX_ROWS:   # stratified subsample keeps the anomaly rate
        idx = np.concatenate([rng.choice(np.where(y == c)[0], int(round(MAX_ROWS * (y == c).mean())), replace=False)
                              for c in (0, 1)])
        X, y = X[np.sort(idx)], y[np.sort(idx)]
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    df["is_anomaly"] = y
    name = f.stem.split("_", 1)[1] + ".csv"
    df.to_csv(out / name, index=False)
    print(f"{name:28s} {len(df):6d} rows {X.shape[1]:4d} feats  anomaly rate {y.mean():.3f}")


def from_classification(src, name, label_col, is_anomaly, rate=0.10, max_normal=None):
    """ADBench recipe for classification data: one class = normal, the other class(es)
    downsampled to `rate` of the final dataset as anomalies."""
    df = pd.read_csv(src)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    lab = df[label_col]
    df = df.drop(columns=[label_col])
    anom = is_anomaly(lab)
    normal_idx = np.where(~anom)[0]
    if max_normal and len(normal_idx) > max_normal:
        normal_idx = rng.choice(normal_idx, max_normal, replace=False)
    n_anom = int(round(len(normal_idx) * rate / (1 - rate)))
    anom_idx = rng.choice(np.where(anom)[0], min(n_anom, anom.sum()), replace=False)
    idx = np.sort(np.concatenate([normal_idx, anom_idx]))
    df = df.iloc[idx].reset_index(drop=True)
    df["is_anomaly"] = anom[idx].astype(int)
    df.to_csv(out / name, index=False)
    print(f"{name:28s} {len(df):6d} rows {df.shape[1]-1:4d} feats  anomaly rate {df.is_anomaly.mean():.3f}")


old = repo / "Datasets" / "Anamoly"
from_classification(old / "kidney_disease.csv", "kidney_disease.csv", "classification",
                    lambda s: s.astype(str).str.strip().eq("ckd").to_numpy())
from_classification(old / "indian_liver_patient.csv", "indian_liver_patient.csv", "Dataset",
                    lambda s: (s == 1).to_numpy())
mit = pd.read_csv(old / "mitbih_test.csv", header=None)
mit.columns = [f"t{i}" for i in range(mit.shape[1] - 1)] + ["beat_class"]
mit.to_csv(npz_dir / "_mitbih.csv", index=False)
from_classification(npz_dir / "_mitbih.csv", "mitbih_ecg.csv", "beat_class",
                    lambda s: (s != 0).to_numpy(), rate=0.05, max_normal=5000)
