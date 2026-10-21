"""Central configuration: paths, task names, metrics, model registry metadata.

Model names deliberately match the `model` column in Benchmark_results/*_detailed.csv
so the knowledge base, ranker and HPO all speak the same vocabulary.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = REPO_ROOT / "Datasets"
BENCHMARK_DIR = REPO_ROOT / "Benchmark_results"        # original benchmarks from Code/run_*_benchmark.py
ARTIFACTS_DIR = REPO_ROOT / "artifacts"          # knowledge base, reports
ARTIFACTS_DIR.mkdir(exist_ok=True)
KB_PATH = ARTIFACTS_DIR / "knowledge_base.csv"
RUNS_DIR = ARTIFACTS_DIR / "runs"
# Benchmarks re-run by metalearner/benchmark.py; preferred over BENCHMARK_DIR when present
BENCHMARK_V2_DIR = ARTIFACTS_DIR / "benchmarks"

# Parser task labels -> internal task family used by KB / ranker / HPO
TASK_FAMILY = {
    "binary_classification": "classification",
    "multiclass_classification": "classification",
    "classification": "classification",
    "regression": "regression",
    "survival_analysis": "survival",
    "survival": "survival",
    "time_series_forecasting": "timeseries",
    "timeseries": "timeseries",
    "anomaly_detection": "anomaly",
}
PARSER_TASKS = [
    "binary_classification", "multiclass_classification", "regression",
    "survival_analysis", "time_series_forecasting", "anomaly_detection",
]

# Benchmark file + dataset folder per task family
BENCHMARK_FILES = {
    "classification": ("classification_benchmark_detailed.csv", "Classification"),
    "regression": ("regression_benchmark_detailed.csv", "Regression"),
    "survival": ("survival_benchmark_detailed.csv", "Survival_analysis"),
    "timeseries": ("timeseries_benchmark_detailed.csv", "Time_series"),
    "anomaly": ("anomaly_benchmark_detailed.csv", "Anomaly"),
}

# metric name -> (column in detailed benchmark CSV, higher_is_better)
METRICS = {
    "classification": {
        "f1": ("f1_mean", True),
        "accuracy": ("accuracy_mean", True),
        "precision": ("precision_mean", True),
        "recall": ("recall_mean", True),
        "roc_auc": ("roc_auc_mean", True),
    },
    "regression": {
        "rmse": ("rmse_mean", False),
        "mae": ("mae_mean", False),
        "r2": ("r2_mean", True),
        "mape": ("mape_mean", False),
    },
    "survival": {"c_index": ("c_index", True)},
    "timeseries": {
        "rmse": ("rmse", False),
        "mae": ("mae", False),
        "smape": ("smape", False),
        "r2": ("r2", True),
    },
    "anomaly": {
        "pr_auc": ("pr_auc_mean", True),
        "roc_auc": ("roc_auc_mean", True),
        "precision_at_n": ("precision_at_n_mean", True),
    },
}
DEFAULT_METRIC = {"classification": "f1", "regression": "rmse",
                  "survival": "c_index", "timeseries": "rmse",
                  "anomaly": "pr_auc"}

# How easy a fitted model is to explain to a clinician / stakeholder (0..1).
INTERPRETABILITY = {
    "LogisticRegression": 1.0, "LinearRegression": 1.0, "Ridge": 1.0, "Lasso": 1.0,
    "CoxPH": 1.0, "ARIMA": 0.9, "SARIMA": 0.9, "Prophet": 0.8,
    "KNN": 0.6, "RandomForest": 0.4, "RandomSurvivalForest": 0.4,
    "GradientBoosting": 0.3, "GradientBoostingSurvival": 0.3,
    "XGBoost": 0.3, "CatBoost": 0.3, "SVC": 0.3, "SVR": 0.3,
    # anomaly detectors: per-feature scores (ECOD/HBOS) say which values are unusual
    "ECOD": 0.9, "HBOS": 0.9, "Mahalanobis": 0.7,
    "LOF": 0.5, "IsolationForest": 0.4, "OneClassSVM": 0.2,
}

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
RANDOM_STATE = 42
# threads per model fit; the benchmark lowers it when it runs several datasets in parallel
N_JOBS = int(os.environ.get("METALEARNER_N_JOBS", "-1"))
