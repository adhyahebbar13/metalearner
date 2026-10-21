"""Benchmark every model of a task family on every dataset, with the same leakage-free CV and
Optuna search the pipeline uses, and write the knowledge base's raw material:

    artifacts/benchmarks/<family>_benchmark_detailed.csv
        dataset, model, best_params, training_time, <metric columns of config.METRICS[family]>, ...

Each model is tuned on the family's default metric (config.DEFAULT_METRIC); every other metric is
then measured for that configuration. Per-dataset results are cached under
artifacts/benchmarks/_parts/, so an interrupted run resumes where it stopped.

    python -m metalearner.benchmark --family anomaly --trials 15 --jobs 4
    python -m metalearner.benchmark --family survival --datasets veteran.csv lung.csv
    python -m metalearner.knowledge_base      # then rebuild the KB
"""
from __future__ import annotations

import argparse
import os
import time
import warnings

import pandas as pd

from .config import BENCHMARK_FILES, BENCHMARK_V2_DIR, DATASETS_DIR, DEFAULT_METRIC, METRICS
from .data_prep import prepare
from .dataset_specs import skipped
from .hpo import tune
from .knowledge_base import load_benchmark_dataset
from .models import REGISTRY, is_runnable

warnings.filterwarnings("ignore")


def _part_path(family: str, dataset: str):
    return BENCHMARK_V2_DIR / "_parts" / family / f"{dataset}.csv"


def benchmark_dataset(family: str, dataset: str, n_trials: int = 15, model_timeout: float = 300,
                      force: bool = False) -> pd.DataFrame:
    part = _part_path(family, dataset)
    if part.exists() and not force:
        return pd.read_csv(part)
    warnings.filterwarnings("ignore")   # again: joblib workers start with fresh filters
    df, spec = load_benchmark_dataset(family, dataset)
    spec.primary_metric = DEFAULT_METRIC[family]
    prep = prepare(df, spec)
    col_of = {m: col for m, (col, _) in METRICS[family].items()}
    rows = []
    t_ds = time.time()
    for model in REGISTRY[family]:
        ok, why = is_runnable(family, model)
        if not ok:
            continue
        t0 = time.time()
        row = {"dataset": dataset, "model": model}
        try:
            res = tune(prep, spec, [model], kb=None, n_trials=n_trials, timeout=model_timeout,
                       backend="auto", verbose=False)
            r = res["models"][model]
            if "best_params" not in r:
                raise RuntimeError(r.get("error", "failed"))
            scores = res["_evaluator"].all_metrics(model, r["best_params"])
            row.update(best_params=repr(r["best_params"]), training_time=round(scores.pop("fit_time"), 4),
                       tuned_metric=res["metric"], default_score=r["default_cv_score"], cv_std=r["cv_std"],
                       n_trials=r["n_trials"], tune_seconds=round(time.time() - t0, 1))
            row.update({col_of[k]: v for k, v in scores.items() if k in col_of})
        except Exception as exc:
            row.update(error=f"{type(exc).__name__}: {str(exc)[:200]}", tune_seconds=round(time.time() - t0, 1))
        rows.append(row)
    out = pd.DataFrame(rows)
    out["n_rows"], out["n_features"] = len(prep.X), prep.X.shape[1]
    part.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(part, index=False)
    primary = col_of[DEFAULT_METRIC[family]]
    ok = out.dropna(subset=[primary]) if primary in out else out.iloc[:0]
    best = ok.loc[ok[primary].idxmax() if METRICS[family][DEFAULT_METRIC[family]][1] else ok[primary].idxmin()] \
        if len(ok) else None
    print(f"  {family:12s} {dataset:40s} {len(prep.X):6d} rows  {time.time() - t_ds:6.0f}s  "
          + (f"best {best.model} {primary}={best[primary]:.3f}" if best is not None else "ALL FAILED"), flush=True)
    return out


def run(family: str, datasets: list[str] | None = None, n_trials: int = 15, jobs: int = 1,
        force: bool = False) -> pd.DataFrame:
    folder = DATASETS_DIR / BENCHMARK_FILES[family][1]
    names = datasets or sorted(p.name for p in folder.glob("*.csv"))
    for n in [n for n in names if skipped(family, n)]:
        print(f"  skip {n}: {skipped(family, n)}")
    names = [n for n in names if not skipped(family, n)]
    print(f"Benchmarking {len(names)} {family} datasets x {len(REGISTRY[family])} models, {n_trials} trials each")
    if jobs > 1:
        from joblib import Parallel, delayed
        # split the cores between workers: models otherwise use all of them each (load 80+ on 8 cores)
        per = str(max(1, (os.cpu_count() or 2) // jobs))
        os.environ["METALEARNER_N_JOBS"] = per
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[var] = per
        parts = Parallel(n_jobs=jobs, backend="loky")(
            delayed(benchmark_dataset)(family, n, n_trials, force=force) for n in names)
    else:
        parts = [benchmark_dataset(family, n, n_trials, force=force) for n in names]
    # the detailed file always reflects every cached part of the family, not just this call's datasets
    allparts = [pd.read_csv(p) for p in sorted((BENCHMARK_V2_DIR / "_parts" / family).glob("*.csv"))
                if p.stem in {p2.name for p2 in folder.glob("*.csv")} and not skipped(family, p.stem)]
    detailed = pd.concat(allparts or parts, ignore_index=True)
    path = BENCHMARK_V2_DIR / BENCHMARK_FILES[family][0]
    detailed.to_csv(path, index=False)
    n_err = int(detailed["error"].notna().sum()) if "error" in detailed else 0
    print(f"\n{len(detailed)} rows ({detailed.dataset.nunique()} datasets, {n_err} failed model runs) -> {path}")
    return detailed


def main():
    ap = argparse.ArgumentParser(description="Re-run benchmarks with the meta-learner's CV + HPO")
    ap.add_argument("--family", required=True, choices=list(BENCHMARK_FILES))
    ap.add_argument("--datasets", nargs="*", help="file names inside the family's Datasets/ folder (default: all)")
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--jobs", type=int, default=1, help="datasets benchmarked in parallel")
    ap.add_argument("--force", action="store_true", help="ignore cached per-dataset results")
    a = ap.parse_args()
    run(a.family, a.datasets, a.trials, a.jobs, a.force)


if __name__ == "__main__":
    main()
