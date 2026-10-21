"""Dataset meta-features: simple, statistical, information-theoretic and landmarkers.

The original extractor only produced 5 shape statistics, which cannot tell a
"linear" dataset from a "tree-friendly" one. Landmarkers - cross-validated scores
of very cheap models (stump, 1-NN, naive Bayes, ridge ...) - are consistently the
most informative meta-features in the algorithm-selection literature, and the
gaps between them (e.g. tree vs linear) directly hint at which family will win.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .anomaly import anomaly_meta_features
from .config import RANDOM_STATE
from .data_prep import Prepared, simple_numeric_matrix

MAX_ROWS = 2000  # landmarkers / MI on a subsample keeps extraction to a few seconds


def _safe(fn, default=np.nan):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v = fn()
        v = float(v)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def harrell_c(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    """Harrell's concordance index (higher risk should mean earlier event)."""
    t, e, r = map(np.asarray, (time, event, risk))
    order = np.argsort(t)
    t, e, r = t[order], e[order], r[order]
    conc = ties = total = 0.0
    for i in np.where(e)[0]:
        later = t > t[i]
        if not later.any():
            continue
        total += later.sum()
        conc += (r[i] > r[later]).sum()
        ties += (r[i] == r[later]).sum()
    return (conc + 0.5 * ties) / total if total else np.nan


def _general(prep: Prepared, Xm: np.ndarray) -> dict:
    X = prep.X
    n, d = X.shape
    mf = {
        "log_n_rows": np.log1p(n),
        "log_n_features": np.log1p(d),
        "log_rows_per_feature": np.log1p(n / max(d, 1)),
        "cat_ratio": len(prep.categorical_cols) / max(d, 1),
        "missing_ratio": float(X.isna().mean().mean()) if d else 0.0,
        "max_col_missing": float(X.isna().mean().max()) if d else 0.0,
        "encoded_dim_ratio": Xm.shape[1] / max(d, 1),
    }
    num = X[prep.numeric_cols].select_dtypes(include=[np.number])
    if num.shape[1]:
        vals = num.to_numpy(dtype=float)
        mf["mean_abs_skew"] = _safe(lambda: np.nanmean(np.abs(stats.skew(vals, nan_policy="omit"))))
        mf["mean_kurtosis"] = _safe(lambda: np.clip(np.nanmean(stats.kurtosis(vals, nan_policy="omit")), -5, 50))
        z = (vals - np.nanmean(vals, 0)) / (np.nanstd(vals, 0) + 1e-12)
        mf["outlier_ratio"] = _safe(lambda: np.nanmean(np.abs(z) > 3))
        if num.shape[1] > 1:
            c = num.corr().abs().to_numpy()
            mf["mean_abs_feature_corr"] = _safe(lambda: np.nanmean(c[np.triu_indices_from(c, 1)]))
        else:
            mf["mean_abs_feature_corr"] = 0.0
    else:
        mf.update(mean_abs_skew=0.0, mean_kurtosis=0.0, outlier_ratio=0.0, mean_abs_feature_corr=0.0)
    if Xm.shape[1] > 1:
        mf["pca_95_ratio"] = _safe(lambda: (np.searchsorted(
            np.cumsum(PCA().fit(Xm[:MAX_ROWS]).explained_variance_ratio_), 0.95) + 1) / Xm.shape[1])
    else:
        mf["pca_95_ratio"] = 1.0
    return mf


def _subsample(Xm, y, rng, strat=None):
    if len(Xm) <= MAX_ROWS:
        return Xm, y
    idx = rng.choice(len(Xm), MAX_ROWS, replace=False)
    return Xm[idx], y[idx]


def _classification(prep: Prepared, Xm, rng) -> dict:
    y = np.asarray(prep.y)
    counts = np.bincount(y)
    counts = counts[counts > 0]
    p = counts / counts.sum()
    mf = {
        "n_classes": len(counts),
        "class_entropy": float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0,
        "minority_ratio": float(p.min() / p.max()),
    }
    Xs, ys = _subsample(Xm, y, rng)
    if Xs.shape[1]:
        mi = mutual_info_classif(Xs, ys, random_state=RANDOM_STATE)
        mf["mean_mutual_info"], mf["max_mutual_info"] = float(mi.mean()), float(mi.max())
    n_splits = int(min(3, np.bincount(ys).min())) if np.bincount(ys).min() >= 2 else 2
    cv = StratifiedKFold(n_splits=max(2, n_splits), shuffle=True, random_state=RANDOM_STATE)
    lm = {
        "lm_stump": DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE),
        "lm_tree4": DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
        "lm_1nn": KNeighborsClassifier(n_neighbors=1),
        "lm_nb": GaussianNB(),
        "lm_lda": LinearDiscriminantAnalysis(),
    }
    for k, est in lm.items():
        mf[k] = _safe(lambda est=est: cross_val_score(est, Xs, ys, cv=cv, scoring="balanced_accuracy").mean())
    mf["lm_gap_tree_vs_linear"] = mf["lm_tree4"] - mf["lm_lda"]
    mf["lm_gap_knn_vs_linear"] = mf["lm_1nn"] - mf["lm_lda"]
    return mf


def _regression(prep: Prepared, Xm, rng) -> dict:
    y = np.asarray(prep.y, dtype=float)
    mf = {
        "target_abs_skew": _safe(lambda: abs(stats.skew(y))),
        "target_kurtosis": _safe(lambda: np.clip(stats.kurtosis(y), -5, 50)),
        "target_cv": _safe(lambda: np.std(y) / (abs(np.mean(y)) + 1e-12)),
        "target_n_unique_ratio": len(np.unique(y)) / len(y),
    }
    Xs, ys = _subsample(Xm, y, rng)
    if Xs.shape[1]:
        corr = np.abs([np.corrcoef(Xs[:, j], ys)[0, 1] if Xs[:, j].std() > 0 else 0 for j in range(Xs.shape[1])])
        mf["mean_abs_target_corr"], mf["max_abs_target_corr"] = float(np.nanmean(corr)), float(np.nanmax(corr))
        mi = _safe(lambda: mutual_info_regression(Xs, ys, random_state=RANDOM_STATE).max())
        mf["max_mutual_info"] = mi
    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    lm = {
        "lm_ridge": Ridge(alpha=1.0),
        "lm_tree3": DecisionTreeRegressor(max_depth=3, random_state=RANDOM_STATE),
        "lm_5nn": KNeighborsRegressor(n_neighbors=5),
    }
    for k, est in lm.items():
        mf[k] = _safe(lambda est=est: np.clip(cross_val_score(est, Xs, ys, cv=cv, scoring="r2").mean(), -1, 1))
    mf["lm_gap_tree_vs_linear"] = mf["lm_tree3"] - mf["lm_ridge"]
    mf["lm_gap_knn_vs_linear"] = mf["lm_5nn"] - mf["lm_ridge"]
    return mf


def _survival(prep: Prepared, Xm, rng) -> dict:
    ev, t = prep.y["event"], prep.y["time"]
    mf = {
        "event_rate": float(ev.mean()),
        "log_median_time": float(np.log1p(np.median(t))),
        "time_cv": float(np.std(t) / (np.mean(t) + 1e-12)),
        "events_per_feature": float(np.log1p(ev.sum() / max(Xm.shape[1], 1))),
    }
    idx = rng.choice(len(t), min(len(t), 1000), replace=False)
    Xs, ts, es = Xm[idx], t[idx], ev[idx]
    logt = np.log1p(ts)
    if Xs.shape[1] and es.sum() > 3:
        corr = [abs(np.corrcoef(Xs[es, j], logt[es])[0, 1]) if Xs[es, j].std() > 0 else 0 for j in range(Xs.shape[1])]
        mf["max_abs_time_corr"] = float(np.nanmax(corr))
        # landmarkers: in-sample-free 2-fold concordance of a linear and a shallow tree model on log time
        folds = np.arange(len(ts)) % 2
        for name, mk in {"lm_linear_cindex": lambda: Ridge(alpha=1.0),
                         "lm_tree_cindex": lambda: DecisionTreeRegressor(max_depth=3, random_state=RANDOM_STATE)}.items():
            scores = []
            for f in (0, 1):
                tr, te = folds != f, folds == f
                m = mk().fit(Xs[tr & es], logt[tr & es]) if (tr & es).sum() > 5 else None
                if m is not None:
                    scores.append(harrell_c(ts[te], es[te], -m.predict(Xs[te])))
            mf[name] = float(np.nanmean(scores)) if scores else np.nan
        mf["lm_gap_tree_vs_linear"] = mf["lm_tree_cindex"] - mf["lm_linear_cindex"]
    return mf


def _timeseries(prep: Prepared, Xm, rng) -> dict:
    y = np.asarray(prep.y, dtype=float)
    n = len(y)
    sd = y.std() + 1e-12

    def acf(k):
        return np.corrcoef(y[:-k], y[k:])[0, 1] if n > k + 2 else np.nan

    mf = {
        "acf1": _safe(lambda: acf(1)),
        "acf_max_2_24": _safe(lambda: np.nanmax([acf(k) for k in range(2, min(25, n // 3))])),
        "trend_r2": _safe(lambda: np.corrcoef(np.arange(n), y)[0, 1] ** 2),
        "target_cv": _safe(lambda: sd / (abs(y.mean()) + 1e-12)),
        "diff_std_ratio": _safe(lambda: np.std(np.diff(y)) / sd),
        "n_exog": prep.X.shape[1],
    }
    split = int(n * 0.8)
    if split > 10 and n - split > 2:
        naive = np.sqrt(np.mean((y[split:] - y[split - 1:-1]) ** 2)) / sd
        mf["lm_naive_nrmse"] = float(naive)
        L = 5
        lag = np.column_stack([y[L - k - 1:n - k - 1] for k in range(L)])
        yt = y[L:]
        s = split - L
        for name, est in {"lm_ridge_lag_nrmse": Ridge(1.0),
                          "lm_tree_lag_nrmse": DecisionTreeRegressor(max_depth=4, random_state=RANDOM_STATE)}.items():
            mf[name] = _safe(lambda est=est: np.sqrt(np.mean((est.fit(lag[:s], yt[:s]).predict(lag[s:]) - yt[s:]) ** 2)) / sd)
        mf["lm_gap_tree_vs_linear"] = mf.get("lm_ridge_lag_nrmse", np.nan) - mf.get("lm_tree_lag_nrmse", np.nan)
    return mf


def _anomaly(prep: Prepared, Xm, rng) -> dict:
    return anomaly_meta_features(Xm, rng, MAX_ROWS)   # label-free on purpose


def compute_meta_features(prep: Prepared) -> dict:
    rng = np.random.default_rng(RANDOM_STATE)
    Xm = simple_numeric_matrix(prep) if prep.X.shape[1] else np.zeros((len(prep.X), 0))
    mf = _general(prep, Xm)
    fn = {"classification": _classification, "regression": _regression,
          "survival": _survival, "timeseries": _timeseries, "anomaly": _anomaly}[prep.family]
    if Xm.shape[1] or prep.family == "timeseries":
        mf.update(fn(prep, Xm, rng))
    return {k: (float(v) if v is not None else np.nan) for k, v in mf.items()}
