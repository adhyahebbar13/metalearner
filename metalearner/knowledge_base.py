"""Build the meta-knowledge base: (dataset meta-features) x (model performance, best params).

Reads Benchmark_results/*_detailed.csv produced by Code/run_*_benchmark.py and the
matching CSVs under Datasets/. Targets follow the benchmark scripts' convention
(last column; survival time/event via the same keyword heuristic) so that the
meta-features describe exactly the problem each benchmark score was measured on.

Usage:  python -m metalearner.knowledge_base
"""
from __future__ import annotations

import ast
import time
import warnings

import numpy as np
import pandas as pd

from .config import BENCHMARK_DIR, BENCHMARK_FILES, BENCHMARK_V2_DIR, DATASETS_DIR, KB_PATH, METRICS
from .data_prep import load_csv, prepare
from .dataset_specs import override, skipped
from .intent import TaskSpec, find_survival_columns, id_like_columns
from .meta_features import compute_meta_features

warnings.filterwarnings("ignore")

_FAMILY_TO_TASK = {"classification": "binary_classification", "regression": "regression",
                   "survival": "survival_analysis", "timeseries": "time_series_forecasting",
                   "anomaly": "anomaly_detection"}


def benchmark_spec(df: pd.DataFrame, family: str, dataset: str | None = None) -> TaskSpec:
    """Task definition for a benchmark dataset: hand-checked overrides (dataset_specs.py) first,
    then the same heuristics a user's data gets; identifier columns are always excluded."""
    spec = _heuristic_spec(df, family)
    ov = override(family, dataset) if dataset else {}
    if family == "survival":
        spec.time_column = ov.get("time", spec.time_column)
        spec.event_column = spec.target = ov.get("event", spec.event_column)
        if "event_value" in ov:
            spec.positive_class = str(ov["event_value"])
    elif "target" in ov:
        spec.target = ov["target"]
    keep = (spec.target, spec.time_column, spec.event_column)
    spec.exclude_columns = list(dict.fromkeys([*spec.exclude_columns, *ov.get("exclude", []),
                                               *id_like_columns(df, keep=keep)]))
    spec.exclude_columns = [c for c in spec.exclude_columns if c in df.columns and c not in keep]
    return spec


def load_benchmark_dataset(family: str, dataset: str) -> tuple[pd.DataFrame, TaskSpec]:
    df = load_csv(DATASETS_DIR / BENCHMARK_FILES[family][1] / dataset)
    q = override(family, dataset).get("query")
    if q:
        df = df.query(q).reset_index(drop=True)
    return df, benchmark_spec(df, family, dataset)


def _heuristic_spec(df: pd.DataFrame, family: str) -> TaskSpec:
    task = _FAMILY_TO_TASK[family]
    if family == "survival":
        t, e = find_survival_columns(df)
        return TaskSpec(task=task, target=e, time_column=t, event_column=e)
    if family == "timeseries":
        num = df.select_dtypes(include=[np.number])
        return TaskSpec(task=task, target=df.columns[-1],
                        exclude_columns=[c for c in df.columns if c not in num.columns and c != df.columns[-1]])
    if family == "anomaly":   # Datasets/Anomaly files end with an is_anomaly 0/1 column
        return TaskSpec(task=task, target=df.columns[-1], positive_class="1")
    return TaskSpec(task=task, target=df.columns[-1])


def benchmark_path(family: str):
    """metalearner/benchmark.py output if it exists, else the original Benchmark_results file."""
    name = BENCHMARK_FILES[family][0]
    return BENCHMARK_V2_DIR / name if (BENCHMARK_V2_DIR / name).exists() else BENCHMARK_DIR / name


def _parse_params(s) -> dict:
    if not isinstance(s, str) or s.strip() in ("", "default", "nan"):
        return {}
    try:
        return ast.literal_eval(s)
    except Exception:
        return {}


def build(verbose: bool = True) -> pd.DataFrame:
    rows = []
    for family, (bench_file, folder) in BENCHMARK_FILES.items():
        path = benchmark_path(family)
        if not path.exists():
            if verbose:
                print(f"  skip {family}: no {bench_file}")
            continue
        bench = pd.read_csv(path)
        if verbose:
            print(f"  {family}: {path.relative_to(path.parents[1])}")
        metric_cols = [col for col, _ in METRICS[family].values() if col in bench.columns]
        if "error" in bench:
            bench = bench[bench["error"].isna()]
        for ds_name, grp in bench.groupby("dataset"):
            path = DATASETS_DIR / folder / ds_name
            why = skipped(family, ds_name)
            if why or not path.exists():
                if verbose:
                    print(f"  skip {family}/{ds_name}: {why or 'file not in Datasets/'}")
                continue
            t0 = time.time()
            try:
                df, spec = load_benchmark_dataset(family, ds_name)
                prep = prepare(df, spec)
                mf = compute_meta_features(prep)
            except Exception as exc:
                print(f"  FAIL {family}/{ds_name}: {exc}")
                continue
            if verbose:
                print(f"  {family:14s} {ds_name:45s} {len(prep.X):6d} rows  {len(mf):2d} meta-features  {time.time()-t0:.1f}s")
            for _, r in grp.iterrows():
                row = {"family": family, "dataset": ds_name, "model": r["model"],
                       "best_params": repr(_parse_params(r.get("best_params"))),
                       "training_time": pd.to_numeric(r.get("training_time"), errors="coerce")}
                for c in metric_cols:
                    row[f"perf__{c}"] = pd.to_numeric(r[c], errors="coerce")
                row.update({f"mf__{k}": v for k, v in mf.items()})
                rows.append(row)
    kb = pd.DataFrame(rows)
    kb.to_csv(KB_PATH, index=False)
    if verbose:
        print(f"\nKnowledge base: {len(kb)} rows, {kb.dataset.nunique()} datasets -> {KB_PATH}")
    return kb


def load_kb() -> pd.DataFrame:
    if not KB_PATH.exists():
        return build()
    return pd.read_csv(KB_PATH)


if __name__ == "__main__":
    build()
