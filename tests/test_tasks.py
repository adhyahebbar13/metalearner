"""Anomaly detection, survival column detection and leakage guards.
Run: python -m pytest tests -q   (or: python tests/test_tasks.py)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metalearner.anomaly import binarize_labels, precision_at_n  # noqa: E402
from metalearner.data_prep import prepare  # noqa: E402
from metalearner.hpo import cross_fit_scores, tune  # noqa: E402
from metalearner.intent import find_survival_columns, id_like_columns, parse_intent  # noqa: E402
from metalearner.models import REGISTRY  # noqa: E402

RNG = np.random.default_rng(0)


def _anomaly_df(labelled=True):
    X = np.vstack([RNG.normal(size=(300, 4)), RNG.uniform(-8, 8, size=(15, 4))])   # scattered outliers
    df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    if labelled:
        df["is_anomaly"] = np.r_[np.zeros(300, int), np.ones(15, int)]
    return df


def test_detectors_rank_scattered_outliers_high():
    df = _anomaly_df()
    X, y = df.drop(columns="is_anomaly").to_numpy(), df.is_anomaly.to_numpy()
    for name, entry in REGISTRY["anomaly"].items():
        s = entry["build"](entry["default"], None).fit(X).decision_function(X)
        assert roc_auc_score(y, s) > 0.9, name


def test_metrics_and_label_binarisation():
    assert precision_at_n([0, 0, 1, 1], [0.1, 0.2, 0.9, 0.8]) == 1.0
    assert precision_at_n([0, 0, 1, 1], [0.9, 0.8, 0.1, 0.2]) == 0.0
    y, rule = binarize_labels(pd.Series(["ok"] * 8 + ["bad"] * 2))
    assert y.sum() == 2 and "minority" in rule
    y, _ = binarize_labels(pd.Series(["N", "N", "N", "S", "V"]))       # multiclass: majority = normal
    assert y.tolist() == [0, 0, 0, 1, 1]


def test_parser_anomaly_labelled_and_unlabelled():
    spec = parse_intent("flag unusual readings, about 5% are outliers", _anomaly_df(), api_key=None, verbose=False)
    assert spec.family == "anomaly" and spec.target == "is_anomaly" and spec.expected_anomaly_rate == 0.05
    spec = parse_intent("find anomalies in these sensor readings", _anomaly_df(False), api_key=None, verbose=False)
    assert spec.family == "anomaly" and spec.target is None
    assert spec.primary_metric == "pr_auc"


def test_anomaly_tuning_and_unsupervised_scores():
    df = _anomaly_df()
    spec = parse_intent("detect anomalies", df, api_key=None, verbose=False)
    prep = prepare(df, spec)
    res = tune(prep, spec, ["IsolationForest", "ECOD"], n_trials=4, backend="random", verbose=False)
    assert res["models"]["ECOD"]["n_trials"] == 0            # parameter-free: default run only
    assert res["models"]["IsolationForest"]["cv_score"] > 0.5
    assert set(res["models"]["ECOD"]["other_metrics"]) == {"roc_auc", "pr_auc", "precision_at_n"}
    s = cross_fit_scores(prepare(_anomaly_df(False), spec.__class__(task="anomaly_detection")), "KNN",
                         REGISTRY["anomaly"]["KNN"]["default"])
    assert s.shape == (315,) and roc_auc_score(df.is_anomaly, s) > 0.9


def test_survival_columns_prefer_exact_and_binary_event():
    n = 60
    df = pd.DataFrame({"patient_id": range(n), "er_status_measured_by_ihc": ["Pos", "Neg"] * 30,
                       "overall_survival_months": RNG.uniform(1, 100, n), "overall_survival": [0, 1] * 30,
                       "pctime": RNG.uniform(1, 50, n), "age": RNG.integers(30, 80, n)})
    assert find_survival_columns(df) == ("overall_survival_months", "overall_survival")
    df2 = pd.DataFrame({"id": range(n), "pctime": RNG.uniform(1, 50, n), "futime": RNG.uniform(1, 99, n),
                        "death": [1, 0, 1] * 20})
    assert find_survival_columns(df2) == ("futime", "death")


def test_id_detection_and_leak_drop():
    n = 80
    t = np.sort(RNG.uniform(1, 100, n))
    df = pd.DataFrame({"Patient (i)": np.arange(1, n + 1), "nottingham_prognostic_index": RNG.normal(4, 1, n),
                       "time": t, "status": RNG.integers(0, 2, n)})
    df["Dead or Alive"] = np.where(df.status == 1, "Dead", "Alive")
    assert id_like_columns(df, keep=("time", "status")) == ["Patient (i)"]
    spec = parse_intent("time until death", df, api_key=None, verbose=False)
    assert "Patient (i)" in spec.exclude_columns
    prep = prepare(df, spec)
    assert "Dead or Alive" not in prep.X.columns and any("leak" in n for n in prep.notes)
    assert "nottingham_prognostic_index" in prep.X.columns


def test_derived_feature_leak_dropped_strong_predictor_kept():
    n = 250
    fat = RNG.uniform(5, 40, n)
    density = 495 / (fat + 450) + RNG.normal(0, 4e-4, n)      # body fat is computed from density (+ noise)
    df = pd.DataFrame({"Density": density, "Abdomen": 70 + fat + RNG.normal(0, 4, n),
                       "Age": RNG.integers(20, 80, n), "BodyFat": fat})
    spec = parse_intent("estimate BodyFat", df, api_key=None, verbose=False)
    prep = prepare(df, spec)
    assert "Density" not in prep.X.columns and "Abdomen" in prep.X.columns


def test_pipeline_unlabelled_anomaly_end_to_end(tmp_path=None):
    from metalearner.pipeline import run
    df = _anomaly_df(labelled=False)
    path = Path(tmp_path or Path(__file__).parent) / "_unlabelled_anomaly.csv"
    df.to_csv(path, index=False)
    try:
        rep = run(str(path), "flag the unusual readings, about 5% are faulty", save=False, verbose=False)
    finally:
        path.unlink()
    out = rep["anomaly_output"]
    assert rep["hpo"] is None and out["n_flagged"] == round(0.05 * len(df))
    assert len(set(out["top_rows"][:15]) & set(range(300, 315))) >= 12   # planted outliers ranked first


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
