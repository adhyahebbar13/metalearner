"""Anomaly detectors, metrics and label-free meta-features.

Every detector is an sklearn-style estimator: `fit(X, y=None)` ignores labels (all detectors
are unsupervised) and `decision_function(X)` returns an anomaly score where HIGHER means MORE
anomalous. That one convention lets them sit at the end of the usual preprocessing Pipeline
and be scored with ranking metrics, so labels are only ever used for evaluation.

ECOD (Li et al., 2022) and HBOS (Goldstein & Dengel, 2012) are implemented here to avoid a
pyod dependency; the rest wrap scikit-learn.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM

from .config import N_JOBS, RANDOM_STATE


def _arr(X) -> np.ndarray:
    return np.asarray(X, dtype=float)


class IForestDetector(BaseEstimator):
    def __init__(self, n_estimators=100, max_samples=256, max_features=1.0):
        self.n_estimators, self.max_samples, self.max_features = n_estimators, max_samples, max_features

    def fit(self, X, y=None):
        X = _arr(X)
        self.model_ = IsolationForest(n_estimators=self.n_estimators, max_samples=min(self.max_samples, len(X)),
                                      max_features=self.max_features, random_state=RANDOM_STATE, n_jobs=N_JOBS).fit(X)
        return self

    def decision_function(self, X):
        return -self.model_.score_samples(_arr(X))


class LOFDetector(BaseEstimator):
    def __init__(self, n_neighbors=20, p=2):
        self.n_neighbors, self.p = n_neighbors, p

    def fit(self, X, y=None):
        X = _arr(X)
        self.model_ = LocalOutlierFactor(n_neighbors=min(self.n_neighbors, len(X) - 1), p=self.p,
                                         novelty=True, n_jobs=N_JOBS).fit(X)
        return self

    def decision_function(self, X):
        return -self.model_.score_samples(_arr(X))


class OCSVMDetector(BaseEstimator):
    """One-class SVM; trained on at most `max_train` rows because the solver is ~O(n^2)."""

    def __init__(self, nu=0.1, gamma="scale", max_train=3000):
        self.nu, self.gamma, self.max_train = nu, gamma, max_train

    def fit(self, X, y=None):
        X = _arr(X)
        if len(X) > self.max_train:
            X = X[np.random.default_rng(RANDOM_STATE).choice(len(X), self.max_train, replace=False)]
        self.model_ = OneClassSVM(nu=self.nu, gamma=self.gamma, kernel="rbf").fit(X)
        return self

    def decision_function(self, X):
        return -self.model_.decision_function(_arr(X))


class KNNDetector(BaseEstimator):
    """Distance to the k-th ("largest") or mean of the k nearest training points."""

    def __init__(self, n_neighbors=5, method="largest"):
        self.n_neighbors, self.method = n_neighbors, method

    def fit(self, X, y=None):
        X = _arr(X)
        self.k_ = min(self.n_neighbors, len(X))
        self.nn_ = NearestNeighbors(n_neighbors=self.k_, n_jobs=N_JOBS).fit(X)
        return self

    def decision_function(self, X):
        d, _ = self.nn_.kneighbors(_arr(X))
        return d[:, -1] if self.method == "largest" else d.mean(axis=1)


class ECODDetector(BaseEstimator):
    """Empirical-CDF outlier detection: sum over features of -log tail probability. Parameter-free."""

    def fit(self, X, y=None):
        X = _arr(X)
        self.sorted_ = np.sort(X, axis=0)
        self.skew_ = np.nan_to_num(stats.skew(X, axis=0))
        return self

    def decision_function(self, X):
        X = _arr(X)
        n = self.sorted_.shape[0]
        left = np.empty_like(X)
        right = np.empty_like(X)
        for j in range(X.shape[1]):
            col = self.sorted_[:, j]
            left[:, j] = (np.searchsorted(col, X[:, j], side="right") + 1) / (n + 1)
            right[:, j] = (n - np.searchsorted(col, X[:, j], side="left") + 1) / (n + 1)
        ol, orr = -np.log(left), -np.log(right)
        auto = np.where(self.skew_ < 0, ol, orr)
        return np.max([ol.sum(1), orr.sum(1), auto.sum(1)], axis=0)


class HBOSDetector(BaseEstimator):
    """Histogram-based outlier score: sum over features of -log(normalised bin height)."""

    def __init__(self, n_bins=10, alpha=0.1):
        self.n_bins, self.alpha = n_bins, alpha

    def fit(self, X, y=None):
        X = _arr(X)
        self.edges_, self.heights_ = [], []
        for j in range(X.shape[1]):
            h, e = np.histogram(X[:, j], bins=self.n_bins)
            h = h / max(h.max(), 1)
            self.edges_.append(e)
            self.heights_.append(h + self.alpha)  # alpha keeps empty bins finite
        return self

    def decision_function(self, X):
        X = _arr(X)
        score = np.zeros(len(X))
        for j, (e, h) in enumerate(zip(self.edges_, self.heights_)):
            b = np.searchsorted(e, X[:, j], side="right") - 1
            inside = (b >= 0) & (b < len(h))
            hb = np.where(inside, h[np.clip(b, 0, len(h) - 1)], self.alpha)   # out of range = emptiest bin
            score += -np.log(hb)
        return score


class MahalanobisDetector(BaseEstimator):
    """Mahalanobis distance under a Ledoit-Wolf shrunk covariance (stable in high dimensions)."""

    def fit(self, X, y=None):
        self.cov_ = LedoitWolf().fit(_arr(X))
        return self

    def decision_function(self, X):
        return self.cov_.mahalanobis(_arr(X))


# ----------------------------------------------------------------------------
# Metrics: all take (y_true 0/1, anomaly score) and are higher-is-better
# ----------------------------------------------------------------------------
def precision_at_n(y, s) -> float:
    """Precision among the n highest-scoring rows, n = number of true anomalies."""
    y = np.asarray(y)
    n = int(y.sum())
    return float(y[np.argsort(-np.asarray(s), kind="stable")[:n]].mean()) if n else np.nan


ANOMALY_METRICS = {"roc_auc": roc_auc_score, "pr_auc": average_precision_score, "precision_at_n": precision_at_n}


def anomaly_scorer(metric: str):
    fn = ANOMALY_METRICS[metric]
    return lambda est, X, y: float(fn(y, est.decision_function(X)))


def binarize_labels(s, positive_class=None) -> tuple[np.ndarray, str]:
    """Known-anomaly labels -> 0/1. Returns (labels, human-readable rule used)."""
    v = s.astype(str).str.strip()
    counts = v.value_counts()
    if positive_class is not None and str(positive_class).strip() in counts.index:
        pos = str(positive_class).strip()
        return (v == pos).to_numpy().astype(int), f"{pos!r} = anomaly"
    if len(counts) == 2:
        pos = counts.index[-1]   # minority class
        return (v == pos).to_numpy().astype(int), f"minority class {pos!r} = anomaly"
    normal = counts.index[0]
    return (v != normal).to_numpy().astype(int), f"most frequent class {normal!r} = normal, all others = anomaly"


# ----------------------------------------------------------------------------
# Label-free meta-features (ranking must work for unlabelled data too)
# ----------------------------------------------------------------------------
def anomaly_meta_features(Xm: np.ndarray, rng, max_rows: int = 2000) -> dict:
    if len(Xm) > max_rows:
        Xm = Xm[rng.choice(len(Xm), max_rows, replace=False)]
    n = len(Xm)
    k = min(10, n - 1)
    d, _ = NearestNeighbors(n_neighbors=k + 1).fit(Xm).kneighbors(Xm)
    knn = d[:, -1]                                            # self excluded by the +1
    lof = -LocalOutlierFactor(n_neighbors=k).fit(Xm).negative_outlier_factor_
    iso = -IsolationForest(n_estimators=100, random_state=RANDOM_STATE).fit(Xm).score_samples(Xm)
    ecod = ECODDetector().fit(Xm).decision_function(Xm)
    maha = MahalanobisDetector().fit(Xm).decision_function(Xm)

    def rho(a, b):
        r = stats.spearmanr(a, b)[0]
        return float(r) if np.isfinite(r) else 0.0

    med = np.median(knn) + 1e-12
    return {
        # agreement between detector families: global vs local vs per-feature anomalies
        "lm_rho_iforest_lof": rho(iso, lof),
        "lm_rho_iforest_knn": rho(iso, knn),
        "lm_rho_lof_knn": rho(lof, knn),
        "lm_rho_ecod_iforest": rho(ecod, iso),
        "lm_rho_maha_iforest": rho(maha, iso),
        # shape of the score distributions
        "lm_iforest_score_skew": float(stats.skew(iso)),
        "lm_knn_tail_ratio": float(np.mean(np.sort(knn)[-max(1, n // 100):]) / med),
        "lm_lof_frac_gt_1_5": float(np.mean(lof > 1.5)),
        "knn_dist_cv": float(np.std(knn) / (np.mean(knn) + 1e-12)),
        "duplicate_ratio": float(np.mean(d[:, 1] < 1e-9)),
    }
