"""Hyper-parameter optimisation for the top-ranked models (CASH restricted to the shortlist).

One search over a conditional space:   model in {top-1, top-2}  ->  that model's hyper-parameters.
  * Optuna TPE sampler when optuna is installed (falls back to seeded random search otherwise),
  * warm-started with the library defaults and with the best_params the benchmarks found on the
    most similar KB datasets (the meta-learner's neighbours), snapped into the search space,
  * every model on the shortlist is guaranteed a minimum share of the budget, the rest goes
    wherever the sampler finds improvement,
  * fold-level median pruning stops hopeless configurations early,
  * all preprocessing is inside the CV pipeline (no leakage).
"""
from __future__ import annotations

import ast
import math
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, f1_score, fbeta_score, get_scorer, make_scorer,
                             mean_absolute_error, mean_absolute_percentage_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from .anomaly import ANOMALY_METRICS, anomaly_scorer
from .config import METRICS, RANDOM_STATE
from .data_prep import Prepared, make_preprocessor
from .forecasting import forecast_metric, make_forecaster, rolling_origin_splits
from .intent import TaskSpec
from .models import REGISTRY, available

warnings.filterwarnings("ignore")

MAX_TUNE_ROWS = 20000
_POSITIVE_WORDS = {"1", "yes", "true", "positive", "y", "ckd", "disease", "sick", "malignant", "m", "dead", "died"}


class TrialPruned(Exception):
    pass


# ----------------------------------------------------------------------------
# Search-space introspection and a dependency-free random "trial"
# ----------------------------------------------------------------------------
class SpaceProbe:
    """Calls a space function once and records each dimension's distribution."""

    def __init__(self):
        self.dims = {}

    def suggest_int(self, name, low, high, log=False):
        self.dims[name] = ("int", low, high, log)
        return low

    def suggest_float(self, name, low, high, log=False):
        self.dims[name] = ("float", low, high, log)
        return low

    def suggest_categorical(self, name, choices):
        self.dims[name] = ("cat", list(choices))
        return choices[0]


class RandomTrial(SpaceProbe):
    def __init__(self, number, rng, fixed=None):
        super().__init__()
        self.number, self.rng, self.fixed, self.params = number, rng, fixed or {}, {}

    def _ret(self, name, v):
        v = self.fixed.get(name, v)
        self.params[name] = v
        return v

    def suggest_int(self, name, low, high, log=False):
        v = int(round(math.exp(self.rng.uniform(math.log(low), math.log(high + 1))) - 0.5)) if log \
            else int(self.rng.integers(low, high + 1))
        return self._ret(name, int(np.clip(v, low, high)))

    def suggest_float(self, name, low, high, log=False):
        v = math.exp(self.rng.uniform(math.log(low), math.log(high))) if log else self.rng.uniform(low, high)
        return self._ret(name, float(v))

    def suggest_categorical(self, name, choices):
        return self._ret(name, choices[self.rng.integers(len(choices))])

    def report(self, value, step):
        pass

    def should_prune(self):
        return False


def _snap(params: dict, dims: dict, prefix: str) -> dict:
    """Map a raw param dict (e.g. KB best_params) into the prefixed, in-range search space."""
    out = {}
    for k, v in params.items():
        name = prefix + k
        if name not in dims:
            continue
        d = dims[name]
        if d[0] == "cat":
            if v in d[1]:
                out[name] = v
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                nums = [c for c in d[1] if isinstance(c, (int, float)) and not isinstance(c, bool)]
                if nums:
                    out[name] = min(nums, key=lambda c: abs(c - v))
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and v is not None:
            v = min(max(v, d[1]), d[2])
            out[name] = int(round(v)) if d[0] == "int" else float(v)
    return out


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def positive_index(classes: list, positive_class: str | None) -> int:
    norm = [str(c).strip().lower() for c in classes]
    if positive_class is not None and str(positive_class).strip().lower() in norm:
        return norm.index(str(positive_class).strip().lower())
    for i, c in enumerate(norm):
        if c in _POSITIVE_WORDS:
            return i
    return len(classes) - 1


def build_pipeline(prep: Prepared, model: str, params: dict) -> Pipeline:
    est = REGISTRY[prep.family][model]["build"](params, None)
    scale = model not in ("RandomForest", "GradientBoosting", "XGBoost", "CatBoost",
                          "RandomSurvivalForest", "GradientBoostingSurvival")
    return Pipeline([("pre", make_preprocessor(prep, scale=scale)), ("model", est)])


def cross_fit_scores(prep: Prepared, model: str, params: dict, n_splits: int = 5) -> np.ndarray:
    """Out-of-fold anomaly scores for every row: each row is scored by a detector that never saw it
    (scoring a point with a detector fitted on it lets e.g. kNN/LOF find the point itself)."""
    s = np.empty(len(prep.X))
    for tr, te in KFold(n_splits, shuffle=True, random_state=RANDOM_STATE).split(prep.X):
        s[te] = build_pipeline(prep, model, params).fit(prep.X.iloc[tr]).decision_function(prep.X.iloc[te])
    return s


class Evaluator:
    """Cross-validated score of (model, params) on the prepared data. Always 'higher is better'
    internally (`sign` flips error metrics); `natural()` converts back for reporting."""

    def __init__(self, prep: Prepared, spec: TaskSpec, rng: np.random.Generator):
        self.prep, self.spec, self.family = prep, spec, prep.family
        self.metric = spec.primary_metric
        self.notes = []
        X, y = prep.X, prep.y
        n = len(X)
        if self.family != "timeseries" and n > MAX_TUNE_ROWS:
            idx = np.sort(rng.choice(n, MAX_TUNE_ROWS, replace=False))
            X, y = X.iloc[idx].reset_index(drop=True), y[idx]
            self.notes.append(f"Tuning on a {MAX_TUNE_ROWS}-row subsample; final models refit on all {n} rows.")
        self.X, self.y = X, y

        if self.family == "classification":
            n_classes = len(np.unique(y))
            self.binary = n_classes == 2
            if self.binary:
                pos = positive_index(prep.classes_, spec.positive_class)
                self.y = (np.asarray(y) == pos).astype(int)
                self.positive_label = prep.classes_[pos]
            avg = "binary" if self.binary else "weighted"
            if self.metric in ("recall", "precision"):
                # Pure recall (precision) is trivially maximised by predicting everything positive
                # (almost nothing positive), so optimise the matching F-beta and report both.
                beta = 2.0 if self.metric == "recall" else 0.5
                self.scorer = make_scorer(fbeta_score, beta=beta, average=avg, zero_division=0)
                self.metric = "f2" if beta == 2.0 else "f0.5"
                self.notes.append(f"Optimising {self.metric} (F-beta weighted towards {spec.primary_metric}) "
                                  f"instead of raw {spec.primary_metric}, which a degenerate model can max out.")
            else:
                name = {"f1": "f1", "accuracy": "accuracy", "roc_auc": "roc_auc"}[self.metric]
                if not self.binary and name == "f1":
                    name = "f1_weighted"
                if not self.binary and name == "roc_auc":
                    name = "roc_auc_ovr_weighted"
                self.scorer = get_scorer(name)
            self.sign = 1.0
            k = int(min(5, np.bincount(self.y).min()))
            self.splits = list(StratifiedKFold(max(2, k), shuffle=True, random_state=RANDOM_STATE).split(self.X, self.y))
        elif self.family == "regression":
            name = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error",
                    "r2": "r2", "mape": "neg_mean_absolute_percentage_error"}[self.metric]
            self.scorer, self.sign = get_scorer(name), 1.0
            self.natural_sign = -1.0 if name.startswith("neg_") else 1.0
            self.splits = list(KFold(5 if n < 5000 else 3, shuffle=True, random_state=RANDOM_STATE).split(self.X))
        elif self.family == "survival":
            ev = y["event"].astype(int)
            k = int(min(5, np.bincount(ev).min())) if ev.min() != ev.max() else 2
            self.splits = list(StratifiedKFold(max(2, k), shuffle=True, random_state=RANDOM_STATE).split(self.X, ev))
            self.sign = 1.0
        elif self.family == "anomaly":
            if y is None:
                raise ValueError("anomaly tuning needs labels; use unsupervised_fit() for unlabelled data")
            self.scorer, self.sign = anomaly_scorer(self.metric), 1.0
            k = int(min(5, y.sum(), len(y) - y.sum()))
            self.splits = list(StratifiedKFold(max(2, k), shuffle=True, random_state=RANDOM_STATE).split(self.X, y))
        else:  # timeseries
            self.horizon = int(spec.forecast_horizon or max(1, min(int(0.1 * n), 50)))
            self.splits = list(rolling_origin_splits(n, self.horizon, 3))
            self.sign = 1.0 if self.metric == "r2" else -1.0

    def natural(self, internal: float) -> float:
        if self.family == "regression":
            return internal * self.natural_sign
        if self.family == "timeseries":
            return internal * self.sign
        return internal

    def pipeline(self, model: str, params: dict):
        return build_pipeline(self.prep, model, params)

    def evaluate(self, model: str, params: dict, trial=None) -> tuple[float, float]:
        scores = []
        for i, (tr, te) in enumerate(self.splits):
            if self.family == "timeseries":
                f = make_forecaster(model, params).fit(self.y[tr])
                s = self.sign * forecast_metric(self.metric, self.y[te], f.predict(len(te)))
            else:
                pipe = self.pipeline(model, params)
                Xtr, Xte = self.X.iloc[tr], self.X.iloc[te]
                pipe.fit(Xtr, self.y[tr])
                s = pipe.score(Xte, self.y[te]) if self.family == "survival" else self.scorer(pipe, Xte, self.y[te])
            if not np.isfinite(s):
                raise ValueError("non-finite score")
            scores.append(float(s))
            if trial is not None:
                trial.report(float(np.mean(scores)), i)
                if trial.should_prune():
                    raise TrialPruned()
        return float(np.mean(scores)), float(np.std(scores))

    def _fold_metrics(self, pipe, Xte, yte) -> dict:
        """Every metric of the family on one fitted fold, in natural units."""
        out = {}
        if self.family == "anomaly":
            s = pipe.decision_function(Xte)
            return {k: float(fn(yte, s)) for k, fn in ANOMALY_METRICS.items()}
        if self.family == "survival":
            return {"c_index": float(pipe.score(Xte, yte))}
        pred = pipe.predict(Xte)
        if self.family == "regression":
            return {"rmse": float(np.sqrt(mean_squared_error(yte, pred))), "mae": float(mean_absolute_error(yte, pred)),
                    "r2": float(r2_score(yte, pred)), "mape": float(mean_absolute_percentage_error(yte, pred))}
        avg = "binary" if self.binary else "weighted"
        out = {"f1": f1_score(yte, pred, average=avg, zero_division=0), "accuracy": accuracy_score(yte, pred),
               "precision": precision_score(yte, pred, average=avg, zero_division=0),
               "recall": recall_score(yte, pred, average=avg, zero_division=0)}
        try:   # AUC needs scores; SVC without probability has decision_function only
            if hasattr(pipe, "predict_proba"):
                proba = pipe.predict_proba(Xte)
                out["roc_auc"] = roc_auc_score(yte, proba[:, 1]) if self.binary else \
                    roc_auc_score(yte, proba, multi_class="ovr", average="weighted", labels=np.unique(self.y))
            elif self.binary:
                out["roc_auc"] = roc_auc_score(yte, pipe.decision_function(Xte))
        except Exception:
            pass
        return {k: float(v) for k, v in out.items()}

    def all_metrics(self, model: str, params: dict) -> dict:
        """Mean CV value of every metric of the family (one fit per fold), plus mean fit time."""
        if self.family == "timeseries":
            per = {m: [] for m in METRICS["timeseries"]}
            t0 = time.time()
            for tr, te in self.splits:
                pred = make_forecaster(model, params).fit(self.y[tr]).predict(len(te))
                for m in per:
                    per[m].append(forecast_metric(m, self.y[te], pred))
            return {m: float(np.mean(v)) for m, v in per.items()} | {"fit_time": (time.time() - t0) / len(self.splits)}
        per, t_fit = {}, []
        for tr, te in self.splits:
            pipe = self.pipeline(model, params)
            t0 = time.time()
            pipe.fit(self.X.iloc[tr], self.y[tr])
            t_fit.append(time.time() - t0)
            for k, v in self._fold_metrics(pipe, self.X.iloc[te], self.y[te]).items():
                per.setdefault(k, []).append(v)
        out = {k: float(np.nanmean(v)) for k, v in per.items() if np.isfinite(np.nanmean(v))}
        return out | {"fit_time": float(np.mean(t_fit))}

    def extra_metrics(self, model: str, params: dict) -> dict:
        """Secondary CV metrics for the report."""
        if self.family not in ("classification", "anomaly"):
            return {}
        try:
            return {k: v for k, v in self.all_metrics(model, params).items() if k != "fit_time"}
        except Exception:
            return {}


    def fit_final(self, model: str, params: dict):
        if self.family == "timeseries":
            return make_forecaster(model, params).fit(self.prep.y)
        y = self.prep.y
        if self.family == "anomaly":
            return self.pipeline(model, params).fit(self.prep.X)
        if self.family == "classification" and self.binary:
            y = (np.asarray(y) == self.prep.classes_.index(self.positive_label)).astype(int)
        return self.pipeline(model, params).fit(self.prep.X, y)


# ----------------------------------------------------------------------------
# Warm starts from the knowledge base
# ----------------------------------------------------------------------------
def kb_warm_starts(kb: pd.DataFrame, family: str, model: str, neighbours: list[str], k: int = 3) -> list[dict]:
    if kb is None or not neighbours:
        return []
    rows = kb[(kb.family == family) & (kb.model == model) & kb.dataset.isin(neighbours[:k])]
    out = []
    for s in rows["best_params"].dropna():
        try:
            p = ast.literal_eval(s)
        except Exception:
            continue
        if p and p not in out:
            out.append(p)
    return out


def neighbour_params(kb: pd.DataFrame | None, family: str, model: str, neighbours: list[str]) -> tuple[dict, str]:
    """Best params of the most similar KB dataset that has them, snapped into the search space;
    library defaults otherwise. Used when there are no labels to tune on."""
    default = dict(REGISTRY[family][model]["default"])
    probe = SpaceProbe()
    REGISTRY[family][model]["space"](probe, "")
    for ds in neighbours:
        for p in kb_warm_starts(kb, family, model, [ds], k=1):
            snapped = _snap(p, probe.dims, "")
            if snapped:
                return default | snapped, f"best params on {ds}"
    return default, "library defaults"


# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------
def tune(prep: Prepared, spec: TaskSpec, models: list[str], kb: pd.DataFrame | None = None,
         neighbours: list[str] | None = None, n_trials: int = 40, timeout: float | None = None,
         min_share: float = 0.3, backend: str = "auto", verbose: bool = True) -> dict:
    fam = prep.family
    rng = np.random.default_rng(RANDOM_STATE)
    ev = Evaluator(prep, spec, rng)
    use_optuna = backend in ("auto", "optuna") and available(["optuna"])
    if backend == "optuna" and not use_optuna:
        raise ImportError("optuna not installed: pip install optuna")

    # dimensions of each model's conditional space (for snapping warm starts)
    dims = {}
    for m in models:
        probe = SpaceProbe()
        REGISTRY[fam][m]["space"](probe, f"{m}__")
        dims[m] = probe.dims

    # 1) library defaults: the "untuned" reference for every shortlisted model
    results = {m: {"default_score": None, "best_score": -np.inf, "best_std": None, "best_params": None,
                   "n_trials": 0, "n_pruned": 0, "n_failed": 0, "warm_starts": 0} for m in models}
    history = []
    for m in models:
        t0 = time.time()
        try:
            mean, std = ev.evaluate(m, REGISTRY[fam][m]["default"])
            results[m].update(default_score=mean, best_score=mean, best_std=std,
                              best_params=dict(REGISTRY[fam][m]["default"]), source="default")
        except Exception as exc:
            results[m]["default_error"] = str(exc)[:200]
        history.append({"model": m, "source": "default", "score": results[m]["default_score"], "time": time.time() - t0})
        if verbose:
            d = results[m]["default_score"]
            print(f"  [default] {m:24s} {ev.metric}={ev.natural(d):.4f}" if d is not None else f"  [default] {m} failed")

    # 2) warm-start queue: KB neighbours' best params, snapped into the space
    queue = []
    for m in models:
        for p in kb_warm_starts(kb, fam, m, neighbours or []):
            snapped = _snap(p, dims[m], f"{m}__")
            if snapped:
                queue.append({"model": m, **snapped})
                results[m]["warm_starts"] += 1

    # parameter-free models (e.g. ECOD, LinearRegression) are fully evaluated by their default run
    searchable = [m for m in models if dims[m]]
    queue = [q for q in queue if q["model"] in searchable]
    if not searchable:
        n_trials = 0

    def space_fn(trial):
        m = trial.suggest_categorical("model", searchable)
        return m, REGISTRY[fam][m]["space"](trial, f"{m}__")

    min_per_model = int(min_share * n_trials / max(len(searchable), 1))
    study = None
    if use_optuna:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE, multivariate=True, group=True),
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
        for q in queue:
            study.enqueue_trial(q, skip_if_exists=True)

    start = time.time()
    for i in range(n_trials):
        if timeout and time.time() - start > timeout:
            break
        counts = {m: results[m]["n_trials"] for m in searchable}
        remaining = n_trials - i
        starving = [m for m in searchable if counts[m] < min_per_model and
                    sum(max(0, min_per_model - counts[x]) for x in searchable) >= remaining]
        fixed = {"model": starving[0]} if starving else {}

        if use_optuna:
            if fixed and not queue:
                study.enqueue_trial(fixed)
            trial = study.ask()
        else:
            fixed = queue.pop(0) if queue else fixed
            trial = RandomTrial(i, rng, fixed)
        if use_optuna and queue:
            queue.pop(0)

        t0 = time.time()
        m = None
        try:
            m, params = space_fn(trial)
            results[m]["n_trials"] += 1
            mean, std = ev.evaluate(m, params, trial)
            if use_optuna:
                study.tell(trial, mean)
            if mean > results[m]["best_score"]:
                results[m].update(best_score=mean, best_std=std, best_params=params, source=f"trial {i}")
            history.append({"model": m, "source": f"trial {i}", "score": mean, "time": time.time() - t0})
            if verbose:
                print(f"  [trial {i:3d}] {m:24s} {ev.metric}={ev.natural(mean):.4f}  "
                      f"(best {m}: {ev.natural(results[m]['best_score']):.4f})")
        except TrialPruned:
            results[m]["n_pruned"] += 1
            if use_optuna:
                import optuna
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        except Exception as exc:
            if m:
                results[m]["n_failed"] += 1
            if use_optuna:
                import optuna
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
            if verbose:
                print(f"  [trial {i:3d}] {m} failed: {str(exc)[:120]}")

    # strip prefixes, convert to natural units
    summary = {}
    for m, r in results.items():
        if r["best_params"] is None:
            summary[m] = {"error": r.get("default_error", "all trials failed")}
            continue
        bp = {k.split("__", 1)[1] if "__" in k else k: v for k, v in r["best_params"].items()}
        summary[m] = {
            "best_params": bp,
            "cv_score": ev.natural(r["best_score"]),
            "cv_std": r["best_std"],
            "default_cv_score": ev.natural(r["default_score"]) if r["default_score"] is not None else None,
            "improvement_over_default": (r["best_score"] - r["default_score"]) if r["default_score"] is not None else None,
            "n_trials": r["n_trials"], "n_pruned": r["n_pruned"], "n_failed": r["n_failed"],
            "kb_warm_starts": r["warm_starts"], "best_found_by": r.get("source"),
            "other_metrics": ev.extra_metrics(m, bp),
        }
    ok = [m for m in summary if "best_params" in summary[m]]
    winner = max(ok, key=lambda m: results[m]["best_score"]) if ok else None
    return {"backend": "optuna-tpe" if use_optuna else "random-search", "metric": ev.metric,
            "higher_is_better": ev.sign > 0 if fam != "regression" else ev.natural_sign > 0,
            "cv_folds": len(ev.splits), "models": summary, "winner": winner,
            "notes": ev.notes, "_evaluator": ev, "_internal_best": {m: results[m]["best_params"] for m in ok},
            "history": history}
