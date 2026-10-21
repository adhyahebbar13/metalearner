"""Model registry: estimator builders + conditional search spaces for every model in the KB.

Each entry defines
  space(trial, prefix)   -> dict of hyper-parameters, using optuna-style trial.suggest_*
  build(params, ctx)     -> unfitted estimator
  default                -> params of the library default (always tried first)
  requires               -> import names that must be available

Names match Benchmark_results so the ranker's output plugs straight in.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import sklearn
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

from .config import N_JOBS, RANDOM_STATE

_SK_NEW_LOGREG = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)


def available(mods: list[str]) -> bool:
    return all(importlib.util.find_spec(m) is not None for m in mods)


def _logreg(p, ctx):
    kw = dict(C=p["C"], l1_ratio=p["l1_ratio"], solver="saga", max_iter=3000,
              class_weight=p["class_weight"], random_state=RANDOM_STATE)
    if not _SK_NEW_LOGREG:
        kw["penalty"] = "elasticnet"
    return LogisticRegression(**kw)


def _xgb(p, ctx, clf):
    from xgboost import XGBClassifier, XGBRegressor
    kw = dict(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"], max_depth=p["max_depth"],
              subsample=p["subsample"], colsample_bytree=p["colsample_bytree"],
              min_child_weight=p["min_child_weight"], reg_lambda=p["reg_lambda"],
              random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=0)
    return XGBClassifier(eval_metric="logloss", **kw) if clf else XGBRegressor(**kw)


def _cat(p, ctx, clf):
    from catboost import CatBoostClassifier, CatBoostRegressor
    kw = dict(iterations=p["iterations"], learning_rate=p["learning_rate"], depth=p["depth"],
              l2_leaf_reg=p["l2_leaf_reg"], random_seed=RANDOM_STATE, verbose=0,
              allow_writing_files=False, thread_count=N_JOBS)
    return CatBoostClassifier(**kw) if clf else CatBoostRegressor(**kw)


# --- shared spaces -------------------------------------------------------------
def s_rf(t, P, clf):
    d = dict(n_estimators=t.suggest_int(P + "n_estimators", 50, 500, log=True),
             max_depth=t.suggest_categorical(P + "max_depth", [None, 4, 8, 12, 20, 30]),
             min_samples_split=t.suggest_int(P + "min_samples_split", 2, 20),
             min_samples_leaf=t.suggest_int(P + "min_samples_leaf", 1, 10),
             max_features=t.suggest_categorical(P + "max_features", ["sqrt", "log2", 0.5, 1.0]))
    if clf:
        d["class_weight"] = t.suggest_categorical(P + "class_weight", [None, "balanced", "balanced_subsample"])
    return d


def s_gb(t, P):
    return dict(n_estimators=t.suggest_int(P + "n_estimators", 50, 500, log=True),
                learning_rate=t.suggest_float(P + "learning_rate", 0.01, 0.3, log=True),
                max_depth=t.suggest_int(P + "max_depth", 2, 8),
                subsample=t.suggest_float(P + "subsample", 0.5, 1.0))


def s_xgb(t, P):
    return dict(n_estimators=t.suggest_int(P + "n_estimators", 50, 600, log=True),
                learning_rate=t.suggest_float(P + "learning_rate", 0.01, 0.3, log=True),
                max_depth=t.suggest_int(P + "max_depth", 2, 10),
                subsample=t.suggest_float(P + "subsample", 0.5, 1.0),
                colsample_bytree=t.suggest_float(P + "colsample_bytree", 0.4, 1.0),
                min_child_weight=t.suggest_float(P + "min_child_weight", 1.0, 10.0, log=True),
                reg_lambda=t.suggest_float(P + "reg_lambda", 1e-3, 10.0, log=True))


def s_cat(t, P):
    return dict(iterations=t.suggest_int(P + "iterations", 50, 600, log=True),
                learning_rate=t.suggest_float(P + "learning_rate", 0.01, 0.3, log=True),
                depth=t.suggest_int(P + "depth", 3, 8),
                l2_leaf_reg=t.suggest_float(P + "l2_leaf_reg", 1.0, 10.0, log=True))


def s_knn(t, P):
    return dict(n_neighbors=t.suggest_int(P + "n_neighbors", 1, 50, log=True),
                weights=t.suggest_categorical(P + "weights", ["uniform", "distance"]),
                p=t.suggest_categorical(P + "p", [1, 2]))


XGB_DEFAULT = dict(n_estimators=100, learning_rate=0.3, max_depth=6, subsample=1.0, colsample_bytree=1.0,
                   min_child_weight=1.0, reg_lambda=1.0)
CAT_DEFAULT = dict(iterations=500, learning_rate=0.05, depth=6, l2_leaf_reg=3.0)
RF_DEFAULT = dict(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features="sqrt")
GB_DEFAULT = dict(n_estimators=100, learning_rate=0.1, max_depth=3, subsample=1.0)
KNN_DEFAULT = dict(n_neighbors=5, weights="uniform", p=2)

REGISTRY: dict[str, dict[str, dict]] = {
    "classification": {
        "LogisticRegression": dict(
            space=lambda t, P: dict(C=t.suggest_float(P + "C", 1e-3, 1e2, log=True),
                                    l1_ratio=t.suggest_float(P + "l1_ratio", 0.0, 1.0),
                                    class_weight=t.suggest_categorical(P + "class_weight", [None, "balanced"])),
            build=_logreg, default=dict(C=1.0, l1_ratio=0.0, class_weight=None), requires=[]),
        "RandomForest": dict(
            space=lambda t, P: s_rf(t, P, True),
            build=lambda p, c: RandomForestClassifier(**p, n_jobs=N_JOBS, random_state=RANDOM_STATE),
            default=RF_DEFAULT | {"class_weight": None}, requires=[]),
        "GradientBoosting": dict(
            space=s_gb, build=lambda p, c: GradientBoostingClassifier(**p, random_state=RANDOM_STATE),
            default=GB_DEFAULT, requires=[]),
        "XGBoost": dict(space=s_xgb, build=lambda p, c: _xgb(p, c, True), default=XGB_DEFAULT, requires=["xgboost"]),
        "CatBoost": dict(space=s_cat, build=lambda p, c: _cat(p, c, True), default=CAT_DEFAULT, requires=["catboost"]),
        "SVC": dict(
            space=lambda t, P: dict(C=t.suggest_float(P + "C", 1e-2, 1e2, log=True),
                                    kernel=t.suggest_categorical(P + "kernel", ["rbf", "linear"]),
                                    gamma=t.suggest_float(P + "gamma", 1e-4, 1.0, log=True),
                                    class_weight=t.suggest_categorical(P + "class_weight", [None, "balanced"])),
            build=lambda p, c: SVC(**p, random_state=RANDOM_STATE),
            default=dict(C=1.0, kernel="rbf", gamma="scale", class_weight=None), requires=[]),
        "KNN": dict(space=s_knn, build=lambda p, c: KNeighborsClassifier(**p), default=KNN_DEFAULT, requires=[]),
    },
    "regression": {
        "LinearRegression": dict(space=lambda t, P: {}, build=lambda p, c: LinearRegression(), default={}, requires=[]),
        "Ridge": dict(space=lambda t, P: dict(alpha=t.suggest_float(P + "alpha", 1e-3, 1e3, log=True)),
                      build=lambda p, c: Ridge(**p), default=dict(alpha=1.0), requires=[]),
        "Lasso": dict(space=lambda t, P: dict(alpha=t.suggest_float(P + "alpha", 1e-4, 1e1, log=True)),
                      build=lambda p, c: Lasso(**p, max_iter=10000), default=dict(alpha=1.0), requires=[]),
        "RandomForest": dict(space=lambda t, P: s_rf(t, P, False),
                             build=lambda p, c: RandomForestRegressor(**p, n_jobs=N_JOBS, random_state=RANDOM_STATE),
                             default=RF_DEFAULT | {"max_features": 1.0}, requires=[]),
        "GradientBoosting": dict(space=s_gb, build=lambda p, c: GradientBoostingRegressor(**p, random_state=RANDOM_STATE),
                                 default=GB_DEFAULT, requires=[]),
        "XGBoost": dict(space=s_xgb, build=lambda p, c: _xgb(p, c, False), default=XGB_DEFAULT, requires=["xgboost"]),
        "CatBoost": dict(space=s_cat, build=lambda p, c: _cat(p, c, False), default=CAT_DEFAULT, requires=["catboost"]),
        "SVR": dict(space=lambda t, P: dict(C=t.suggest_float(P + "C", 1e-2, 1e3, log=True),
                                            epsilon=t.suggest_float(P + "epsilon", 1e-3, 1.0, log=True),
                                            gamma=t.suggest_float(P + "gamma", 1e-4, 1.0, log=True)),
                    build=lambda p, c: SVR(**p), default=dict(C=1.0, epsilon=0.1, gamma="scale"), requires=[]),
        "KNN": dict(space=s_knn, build=lambda p, c: KNeighborsRegressor(**p), default=KNN_DEFAULT, requires=[]),
    },
}


# --- survival (scikit-survival) ----------------------------------------------------
def _surv_build(name):
    def build(p, c):
        if name == "CoxPH":
            from sksurv.linear_model import CoxPHSurvivalAnalysis
            return CoxPHSurvivalAnalysis(alpha=p["alpha"], ties=p["ties"], n_iter=200)
        if name == "RandomSurvivalForest":
            from sksurv.ensemble import RandomSurvivalForest
            return RandomSurvivalForest(**p, n_jobs=N_JOBS, random_state=RANDOM_STATE)
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis
        return GradientBoostingSurvivalAnalysis(**p, random_state=RANDOM_STATE)
    return build


REGISTRY["survival"] = {
    "CoxPH": dict(space=lambda t, P: dict(alpha=t.suggest_float(P + "alpha", 1e-4, 1e2, log=True),
                                          ties=t.suggest_categorical(P + "ties", ["breslow", "efron"])),
                  build=_surv_build("CoxPH"), default=dict(alpha=0.01, ties="breslow"), requires=["sksurv"]),
    "RandomSurvivalForest": dict(
        space=lambda t, P: dict(n_estimators=t.suggest_int(P + "n_estimators", 50, 500, log=True),
                                min_samples_split=t.suggest_int(P + "min_samples_split", 2, 20),
                                min_samples_leaf=t.suggest_int(P + "min_samples_leaf", 1, 15),
                                max_features=t.suggest_categorical(P + "max_features", ["sqrt", "log2", 0.5, None])),
        build=_surv_build("RandomSurvivalForest"),
        default=dict(n_estimators=100, min_samples_split=6, min_samples_leaf=3, max_features="sqrt"),
        requires=["sksurv"]),
    "GradientBoostingSurvival": dict(space=s_gb, build=_surv_build("GradientBoostingSurvival"),
                                     default=GB_DEFAULT, requires=["sksurv"]),
}

# --- time series (forecasters defined in forecasting.py) ---------------------------
REGISTRY["timeseries"] = {
    "XGBoost": dict(space=lambda t, P: dict(n_lags=t.suggest_int(P + "n_lags", 3, 30), **s_xgb(t, P)),
                    build=None, default=dict(n_lags=5, **XGB_DEFAULT), requires=[]),  # falls back to HistGB
    "ARIMA": dict(space=lambda t, P: dict(p=t.suggest_int(P + "p", 0, 5), d=t.suggest_int(P + "d", 0, 2),
                                          q=t.suggest_int(P + "q", 0, 5)),
                  build=None, default=dict(p=1, d=1, q=1), requires=["statsmodels"]),
    "SARIMA": dict(space=lambda t, P: dict(p=t.suggest_int(P + "p", 0, 3), d=t.suggest_int(P + "d", 0, 1),
                                           q=t.suggest_int(P + "q", 0, 3), P=t.suggest_int(P + "P", 0, 1),
                                           D=t.suggest_int(P + "D", 0, 1), Q=t.suggest_int(P + "Q", 0, 1),
                                           s=t.suggest_categorical(P + "s", [7, 12, 24])),
                   build=None, default=dict(p=1, d=1, q=1, P=1, D=1, Q=1, s=12), requires=["statsmodels"]),
    "Prophet": dict(space=lambda t, P: dict(
                        changepoint_prior_scale=t.suggest_float(P + "changepoint_prior_scale", 1e-3, 0.5, log=True),
                        seasonality_prior_scale=t.suggest_float(P + "seasonality_prior_scale", 1e-2, 10.0, log=True),
                        seasonality_mode=t.suggest_categorical(P + "seasonality_mode", ["additive", "multiplicative"])),
                    build=None, default=dict(changepoint_prior_scale=0.05, seasonality_prior_scale=10.0,
                                             seasonality_mode="additive"), requires=["prophet"]),
}


# --- anomaly detection (detectors defined in anomaly.py; all unsupervised) ----------
from .anomaly import (ECODDetector, HBOSDetector, IForestDetector, KNNDetector, LOFDetector,  # noqa: E402
                      MahalanobisDetector, OCSVMDetector)

REGISTRY["anomaly"] = {
    "IsolationForest": dict(
        space=lambda t, P: dict(n_estimators=t.suggest_int(P + "n_estimators", 50, 400, log=True),
                                max_samples=t.suggest_int(P + "max_samples", 64, 2048, log=True),
                                max_features=t.suggest_float(P + "max_features", 0.3, 1.0)),
        build=lambda p, c: IForestDetector(**p), default=dict(n_estimators=100, max_samples=256, max_features=1.0),
        requires=[]),
    "LOF": dict(space=lambda t, P: dict(n_neighbors=t.suggest_int(P + "n_neighbors", 5, 100, log=True),
                                        p=t.suggest_categorical(P + "p", [1, 2])),
                build=lambda p, c: LOFDetector(**p), default=dict(n_neighbors=20, p=2), requires=[]),
    "OneClassSVM": dict(space=lambda t, P: dict(nu=t.suggest_float(P + "nu", 0.01, 0.5, log=True),
                                                gamma=t.suggest_float(P + "gamma", 1e-4, 1.0, log=True)),
                        build=lambda p, c: OCSVMDetector(**p), default=dict(nu=0.1, gamma="scale"), requires=[]),
    "KNN": dict(space=lambda t, P: dict(n_neighbors=t.suggest_int(P + "n_neighbors", 1, 50, log=True),
                                        method=t.suggest_categorical(P + "method", ["largest", "mean"])),
                build=lambda p, c: KNNDetector(**p), default=dict(n_neighbors=5, method="largest"), requires=[]),
    "ECOD": dict(space=lambda t, P: {}, build=lambda p, c: ECODDetector(), default={}, requires=[]),
    "HBOS": dict(space=lambda t, P: dict(n_bins=t.suggest_int(P + "n_bins", 5, 50, log=True),
                                         alpha=t.suggest_float(P + "alpha", 0.01, 0.5, log=True)),
                 build=lambda p, c: HBOSDetector(**p), default=dict(n_bins=10, alpha=0.1), requires=[]),
    "Mahalanobis": dict(space=lambda t, P: {}, build=lambda p, c: MahalanobisDetector(), default={}, requires=[]),
}


def is_runnable(family: str, model: str) -> tuple[bool, str]:
    fam = REGISTRY.get(family, {})
    if model not in fam:
        return False, f"{model} has no search space for {family}"
    req = fam[model]["requires"]
    if not available(req):
        return False, f"{model} needs `pip install {' '.join('scikit-survival' if r == 'sksurv' else r for r in req)}`"
    return True, ""
