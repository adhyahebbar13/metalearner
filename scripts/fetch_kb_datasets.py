"""Download public datasets that widen the meta-learner's knowledge base and write them as CSVs
into the repo's Datasets/<family> folders (target = last column unless noted in dataset_specs.py).

usage: python scripts/fetch_kb_datasets.py . <cache_dir> [classification regression survival timeseries]
(prefix with SSL_CERT_FILE=$(python -m certifi) if Python cannot verify certificates, e.g. python.org macOS builds)
"""
import socket
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
socket.setdefaulttimeout(60)   # a dead OpenML connection otherwise hangs forever
repo, cache = Path(sys.argv[1]), Path(sys.argv[2])
families = set(sys.argv[3:]) or {"classification", "regression", "survival", "timeseries"}
D = repo / "Datasets"
MAX_ROWS = 5000
rng = np.random.default_rng(42)
RD = "https://vincentarelbundock.github.io/Rdatasets/csv"


def cap(df, strat=None):
    if len(df) <= MAX_ROWS:
        return df
    if strat is None:
        return df.sample(MAX_ROWS, random_state=42).sort_index()
    # sample row positions per class (groupby.apply drops the grouping column in newer pandas)
    frac = MAX_ROWS / len(df)
    keep = [i for _, idx in df.groupby(strat, observed=True).indices.items()
            for i in rng.choice(idx, max(1, int(round(len(idx) * frac))), replace=False)]
    return df.iloc[sorted(keep)]


def save(df, folder, name):
    (D / folder).mkdir(exist_ok=True)
    df.to_csv(D / folder / name, index=False)
    print(f"  {folder:18s} {name:45s} {df.shape}")


def rdata(pkg, item):
    f = cache / "rdatasets" / f"{pkg}_{item}.csv"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(f"{RD}/{pkg}/{item}.csv").to_csv(f, index=False)
    df = pd.read_csv(f)
    return df.drop(columns=[c for c in ("rownames",) if c in df.columns])


def openml(did):
    from sklearn.datasets import fetch_openml
    b = fetch_openml(data_id=did, as_frame=True, data_home=str(cache / "openml"), parser="auto")
    df = b.frame.copy()
    tgt = b.target_names[0] if getattr(b, "target_names", None) else b.target.name
    return df[[c for c in df.columns if c != tgt] + [tgt]], b.details["name"]


CLF = [3, 11, 14, 15, 16, 22, 23, 29, 31, 44, 46, 50, 54, 188, 307, 458, 469, 1049, 1050, 1063, 1067, 1068,
       1462, 1464, 1475, 1487, 1489, 1494, 1497, 1501, 6332, 40966, 40975, 40982, 40983, 40984, 40994, 40701,
       182, 28,
       53, 55, 1488, 38, 13, 10, 27, 35]           # medical: heart-statlog, hepatitis, parkinsons, sick, ...
REG = [44956, 44957, 44958, 44959, 44960, 44962, 44963, 44964, 44965, 44966, 44967, 44969, 44970, 44971, 44972,
       44973, 44977, 44978, 44980, 44981, 44983, 44987, 44989, 44990, 44992, 44993, 44994, 45012, 45402]

if "classification" in families:
    for did in CLF:
        try:
            df, name = openml(did)
            save(cap(df, df.columns[-1]), "Classification", f"openml_{name}.csv")
        except Exception as exc:
            print(f"  FAIL openml {did}: {exc}")
    # haberman from the survival folder: 5-year survival status is a classification target
    h = pd.read_csv(D / "Survival_analysis" / "haberman.csv", header=None,
                    names=["age", "operation_year", "positive_nodes", "survival_status"])
    save(h, "Classification", "haberman.csv")

if "regression" in families:
    for did in REG:
        try:
            df, name = openml(did)
            save(cap(df), "Regression", f"openml_{name}.csv")
        except Exception as exc:
            print(f"  FAIL openml {did}: {exc}")

if "survival" in families:
    import sksurv.datasets as sd
    for fn, name in [(sd.load_whas500, "whas500"), (sd.load_gbsg2, "gbsg2"), (sd.load_aids, "aids"),
                     (sd.load_breast_cancer, "breast_cancer_tdm")]:
        X, y = fn()
        ev, tm = y.dtype.names
        X = X.copy()
        X["time"], X["event"] = y[tm].astype(float), y[ev].astype(int)
        save(X, "Survival_analysis", f"sksurv_{name}.csv")
    for pkg, item in [("survival", "nwtco"), ("survival", "flchain"), ("MASS", "Melanoma"),
                      ("KMsurv", "alloauto"), ("KMsurv", "burn"), ("KMsurv", "hodg"), ("KMsurv", "kidtran"),
                      ("KMsurv", "larynx"), ("KMsurv", "pneumon"), ("KMsurv", "std"), ("KMsurv", "tongue")]:
        try:
            df = rdata(pkg, item)
            save(cap(df), "Survival_analysis", f"r_{item}.csv")
        except Exception as exc:
            print(f"  FAIL rdatasets {pkg}/{item}: {exc}")
    c = pd.read_csv(D / "Regression" / "cirrhosis.csv")         # = Mayo PBC trial: a survival dataset
    save(c, "Survival_analysis", "cirrhosis_pbc.csv")

if "timeseries" in families:
    series = [("datasets", "AirPassengers", None), ("datasets", "Nile", None), ("datasets", "lynx", None),
              ("datasets", "sunspot.month", None), ("datasets", "UKDriverDeaths", None), ("datasets", "ldeaths", None),
              ("datasets", "mdeaths", None), ("datasets", "fdeaths", None), ("datasets", "nottem", None),
              ("datasets", "co2", None), ("datasets", "LakeHuron", None), ("datasets", "BJsales", None),
              ("datasets", "discoveries", None), ("datasets", "WWWusage", None), ("datasets", "austres", None),
              ("datasets", "treering", None), ("datasets", "UKgas", None), ("datasets", "USAccDeaths", None),
              ("datasets", "JohnsonJohnson", None), ("datasets", "Seatbelts", "drivers"),
              ("datasets", "sunspots", None), ("datasets", "EuStockMarkets", "DAX"), ("datasets", "lh", None),
              ("datasets", "nhtemp", None),
              ("fpp2", "a10", None), ("fpp2", "h02", None), ("fpp2", "ausbeer", None), ("fpp2", "auscafe", None),
              ("fpp2", "debitcards", None), ("fpp2", "hyndsight", None), ("fpp2", "qauselec", None),
              ("fpp2", "qcement", None), ("fpp2", "sunspotarea", None), ("fpp2", "usmelec", None),
              ("fpp2", "calls", None), ("forecast", "gas", None), ("forecast", "taylor", None),
              ("forecast", "wineind", None), ("forecast", "woolyrnq", None)]
    for pkg, item, col in series:
        try:
            df = rdata(pkg, item)
            v = df[col] if col else df["value"] if "value" in df else df.select_dtypes("number").iloc[:, -1]
            v = pd.to_numeric(v, errors="coerce").dropna().reset_index(drop=True).iloc[-MAX_ROWS:]
            save(pd.DataFrame({"value": v}), "Time_series", f"ts_{item.replace('.', '_')}.csv")
        except Exception as exc:
            print(f"  FAIL rdatasets {pkg}/{item}: {exc}")
    T = D / "Time_series"
    hr = pd.read_csv(T / "heart_rate.csv")                     # 4 independent heart-rate recordings
    for c in hr.columns:
        save(pd.DataFrame({"value": hr[c].dropna()}), "Time_series", f"derived_heart_rate_{c}.csv")
    for g in ["gait1", "gait2", "gait3"]:                      # one accelerometer channel per recording
        s = pd.read_csv(T / f"{g}.csv", usecols=["ankle_vertical"])["ankle_vertical"].dropna()
        save(pd.DataFrame({"value": s.iloc[:MAX_ROWS].reset_index(drop=True)}), "Time_series",
             f"derived_{g}_ankle_vertical.csv")
    e = pd.read_csv(T / "ecg_1d_timeseries_prediction.csv", sep=";")
    save(pd.DataFrame({"value": pd.to_numeric(e["ecg_value"], errors="coerce").dropna().iloc[:MAX_ROWS]}),
         "Time_series", "derived_ecg.csv")
