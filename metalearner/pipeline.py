"""End-to-end: prompt + CSV  ->  intent  ->  meta-features  ->  ranked models  ->  tuned top-2.

    python -m metalearner.pipeline --csv Datasets/Classification/diabetes.csv \
        --prompt "Screen patients for diabetes; we must not miss positive cases" --trials 40
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import spearmanr

from .config import DEFAULT_METRIC, RUNS_DIR
from .data_prep import load_csv, prepare
from .hpo import build_pipeline, cross_fit_scores, neighbour_params, tune
from .intent import parse_intent
from .knowledge_base import load_kb
from .meta_features import compute_meta_features
from .models import is_runnable
from .ranker import MetaRanker


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 5)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def run(csv_path: str, prompt: str, n_trials: int = 40, top_k: int = 2, timeout: float | None = None,
        api_key: str | None = None, exclude_self: bool = False, backend: str = "auto",
        save: bool = True, verbose: bool = True) -> dict:
    t_start = time.time()
    say = print if verbose else (lambda *a, **k: None)
    csv_path = Path(csv_path)
    df = load_csv(csv_path)

    # 1. intent ------------------------------------------------------------------
    spec = parse_intent(prompt, df, api_key=api_key, verbose=verbose)
    say(f"\n[1] Intent ({spec.source}): task={spec.task}  target={spec.target!r}  metric={spec.primary_metric}  "
        f"interpretability={spec.interpretability}  speed={spec.speed}")
    if spec.family == "survival":
        say(f"    survival columns: time={spec.time_column!r} event={spec.event_column!r}")
    for w in spec.warnings:
        say(f"    ! {w}")
    family = spec.family

    # 2. data + meta-features ---------------------------------------------------------
    prep = prepare(df, spec)
    meta = compute_meta_features(prep)
    say(f"[2] Prepared {prep.X.shape[0]} rows x {prep.X.shape[1]} features "
        f"({len(prep.numeric_cols)} numeric, {len(prep.categorical_cols)} categorical); {len(meta)} meta-features")
    for n in prep.notes:
        say(f"    ! {n}")

    # 3. rank ------------------------------------------------------------------------
    kb = load_kb()
    exclude = csv_path.name if exclude_self and csv_path.name in set(kb.dataset) else None
    ranker = MetaRanker(kb)
    ranked = ranker.rank(family, meta, spec.primary_metric, spec.interpretability, spec.speed,
                         exclude_dataset=exclude)
    neigh = ranker.neighbours(family, meta, spec.primary_metric, exclude_dataset=exclude)
    say(f"[3] Meta-learner ranking (most similar KB datasets: "
        f"{', '.join(f'{d} ({w:.2f})' for d, w in neigh[:3])})")
    conf = ranked[0].confidence if ranked else 0.0
    say(f"    neighbour confidence {conf:.0%} - "
        + ("ranking driven by similar datasets" if conf >= 0.5 else
           "no close match in the knowledge base, ranking leans on what wins across all datasets"))
    for i, r in enumerate(ranked, 1):
        exp = f"  expected {spec.primary_metric}~{r.expected_score:.3f}" if r.expected_score is not None else ""
        say(f"    {i}. {r.model:26s} utility={r.utility:+.3f}  predicted regret={r.predicted_regret:.3f}{exp}")

    # 4. pick top-k runnable models ----------------------------------------------------
    shortlist, skipped = [], []
    for r in ranked:
        ok, why = is_runnable(family, r.model)
        if ok:
            shortlist.append(r.model)
        else:
            skipped.append({"model": r.model, "reason": why})
        if len(shortlist) == top_k:
            break
    for s in skipped:
        say(f"    skipped {s['model']}: {s['reason']}")
    if not shortlist:
        say("[4] None of the ranked models can run in this environment - install the packages above.")
        return _jsonable({"dataset": str(csv_path), "prompt": prompt, "intent": spec.to_dict(),
                          "ranking": [r.to_dict() for r in ranked], "skipped_models": skipped, "hpo": None})
    run_dir = RUNS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{csv_path.stem.replace(' ', '_')}"
    if family == "anomaly" and prep.y is None:
        return _run_unsupervised(df, prep, spec, ranked, shortlist, skipped, kb, neigh, meta, csv_path, prompt,
                                 run_dir, save, say, t_start)
    say(f"[4] Tuning {shortlist} with {n_trials} trials")

    # 5. HPO --------------------------------------------------------------------------
    hpo = tune(prep, spec, shortlist, kb=kb, neighbours=[d for d, _ in neigh], n_trials=n_trials,
               timeout=timeout, backend=backend, verbose=verbose)
    ev = hpo["_evaluator"]

    # 6. final fit + report ------------------------------------------------------------------
    fitted = {}
    if save:
        run_dir.mkdir(parents=True, exist_ok=True)
        for m, res in hpo["models"].items():
            if "best_params" in res:
                try:
                    est = ev.fit_final(m, res["best_params"])
                    path = run_dir / f"{m}.joblib"
                    joblib.dump({"model": est, "spec": spec.to_dict(), "params": res["best_params"],
                                 "classes": prep.classes_, "positive_label": getattr(ev, "positive_label", None),
                                 "feature_columns": list(prep.X.columns)}, path)
                    fitted[m] = str(path)
                except Exception as exc:
                    res["final_fit_error"] = str(exc)[:200]

    report = {
        "dataset": str(csv_path), "prompt": prompt, "intent": spec.to_dict(),
        "data": {"rows": int(prep.X.shape[0]), "features": int(prep.X.shape[1]), "notes": prep.notes},
        "meta_features": meta,
        "knowledge_base": {"excluded_self": exclude, "nearest_datasets": neigh[:5]},
        "ranking": [r.to_dict() for r in ranked],
        "skipped_models": skipped,
        "hpo": {k: v for k, v in hpo.items() if k != "history"},
        "positive_label": getattr(ev, "positive_label", None),
        "saved_models": fitted,
        "runtime_sec": time.time() - t_start,
    }
    w = hpo["winner"]
    if family == "anomaly" and w:
        report["anomaly_output"] = _anomaly_scores(df, prep, spec, w, hpo["models"][w]["best_params"],
                                                   run_dir, save, say)
    report = _jsonable(report)
    if save:
        (run_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
        report["report_path"] = str(run_dir / "report.json")

    say("\n" + "=" * 78)
    say(f"RESULT  ({hpo['backend']}, {hpo['cv_folds']}-fold CV, metric={hpo['metric']}"
        f"{', positive class=' + repr(report['positive_label']) if report['positive_label'] is not None else ''})")
    for m, r in hpo["models"].items():
        if "best_params" not in r:
            say(f"  {m}: failed - {r.get('error')}")
            continue
        d = r["default_cv_score"]
        say(f"  {'*' if m == w else ' '} {m:24s} tuned {r['cv_score']:.4f} +/- {r['cv_std']:.4f}"
            f"   (default {d:.4f})" if d is not None else f"  {m:24s} tuned {r['cv_score']:.4f}")
        say(f"      params: {r['best_params']}")
        if r.get("other_metrics"):
            say("      also: " + "  ".join(f"{k}={v:.3f}" for k, v in r["other_metrics"].items()))
    for n in hpo["notes"]:
        say(f"  note: {n}")
    say(f"  Runtime {report['runtime_sec']:.1f}s" + (f"  ->  {report.get('report_path')}" if save else ""))
    return report


def _anomaly_scores(df, prep, spec, model, params, run_dir, save, say) -> dict:
    """Score every row out-of-fold, flag the top `rate` share, write anomaly_scores.csv."""
    scores = cross_fit_scores(prep, model, params)
    labelled = prep.y is not None
    rate = spec.expected_anomaly_rate or (float(np.mean(prep.y)) if labelled else 0.05)
    n_flag = max(1, int(round(rate * len(scores))))
    order = np.argsort(-scores, kind="stable")
    rank = np.empty(len(scores), dtype=int)
    rank[order] = np.arange(1, len(scores) + 1)
    out = df.loc[prep.row_ids].copy()
    out.insert(0, "csv_row", prep.row_ids)
    out["anomaly_score"], out["anomaly_rank"], out["flagged"] = scores, rank, rank <= n_flag
    out = out.sort_values("anomaly_rank")
    threshold = float(scores[order[n_flag - 1]])
    say(f"  Flagged {n_flag} of {len(scores)} rows ({rate:.1%}"
        f"{', from the label rate' if labelled and not spec.expected_anomaly_rate else ''}) with {model}; "
        f"most anomalous CSV rows: {out.csv_row.head(8).tolist()}")
    res = {"model": model, "flag_rate": rate, "n_flagged": n_flag, "threshold": threshold,
           "top_rows": out.csv_row.head(20).tolist()}
    if save:
        run_dir.mkdir(parents=True, exist_ok=True)
        out.to_csv(run_dir / "anomaly_scores.csv", index=False)
        res["scores_path"] = str(run_dir / "anomaly_scores.csv")
        say(f"  Per-row scores -> {res['scores_path']}")
    return res


def _run_unsupervised(df, prep, spec, ranked, shortlist, skipped, kb, neigh, meta, csv_path, prompt,
                      run_dir, save, say, t_start) -> dict:
    """No anomaly labels: nothing to tune against, so use the meta-learner's top pick with the
    hyper-parameters that won on the most similar benchmark dataset."""
    say(f"[4] No labels -> no tuning. Fitting {shortlist} with parameters from similar benchmark datasets")
    nbrs = [d for d, _ in neigh]
    models, all_scores = {}, {}
    for m in shortlist:
        params, src = neighbour_params(kb, "anomaly", m, nbrs)
        models[m] = {"params": params, "params_from": src}
        all_scores[m] = cross_fit_scores(prep, m, params)
        say(f"    {m:16s} params {params}  ({src})")
    if len(shortlist) > 1:
        r = float(spearmanr(all_scores[shortlist[0]], all_scores[shortlist[1]])[0])
        say(f"    rank agreement of the top-2 detectors (Spearman) = {r:.2f}"
            + ("  - they disagree; review flagged rows from both" if r < 0.5 else ""))
        models["agreement_spearman"] = r
    w = shortlist[0]
    say("\n" + "=" * 78)
    say(f"RESULT  (unsupervised, ranked by meta-learner)  ->  {w}")
    out = _anomaly_scores(df, prep, spec, w, models[w]["params"], run_dir, save, say)
    fitted = {}
    if save:
        for m in shortlist:
            path = run_dir / f"{m}.joblib"
            joblib.dump({"model": build_pipeline(prep, m, models[m]["params"]).fit(prep.X), "spec": spec.to_dict(),
                         "params": models[m]["params"], "feature_columns": list(prep.X.columns),
                         "threshold": out["threshold"] if m == w else None}, path)
            fitted[m] = str(path)
    report = _jsonable({
        "dataset": str(csv_path), "prompt": prompt, "intent": spec.to_dict(),
        "data": {"rows": int(prep.X.shape[0]), "features": int(prep.X.shape[1]), "notes": prep.notes},
        "meta_features": meta, "knowledge_base": {"nearest_datasets": neigh[:5]},
        "ranking": [r.to_dict() for r in ranked], "skipped_models": skipped, "hpo": None,
        "unsupervised": models, "anomaly_output": out, "saved_models": fitted,
        "runtime_sec": time.time() - t_start})
    if save:
        (run_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
        report["report_path"] = str(run_dir / "report.json")
    say(f"  Runtime {report['runtime_sec']:.1f}s" + (f"  ->  {report.get('report_path')}" if save else ""))
    return report


def main():
    ap = argparse.ArgumentParser(description="Prompt-driven meta-learning AutoML")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=None, help="seconds for the HPO phase")
    ap.add_argument("--backend", choices=["auto", "optuna", "random"], default="auto")
    ap.add_argument("--exclude-self", action="store_true",
                    help="drop this dataset's own benchmark rows from the KB (honest demo on repo datasets)")
    a = ap.parse_args()
    run(a.csv, a.prompt, a.trials, a.top_k, a.timeout, exclude_self=a.exclude_self, backend=a.backend)


if __name__ == "__main__":
    main()
