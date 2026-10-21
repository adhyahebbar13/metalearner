"""Meta-learner that ranks candidate models for a new dataset + user intent.

Why this design (and not one regressor per algorithm, or a best-model classifier):
the knowledge base has only ~10-13 datasets per task family. A classifier on
"best_model" labels sees ~11 training points and mostly learns the majority class;
per-algorithm regressors on absolute scores mostly learn "how hard is this dataset",
which is identical for every model and says nothing about the ranking.

So we model *relative* performance:
  1. For every KB dataset, convert each model's score on the chosen metric into a
     normalised regret in [0, 1]  (0 = best model on that dataset, 1 = worst).
  2. For a new dataset, weight KB datasets by meta-feature similarity
     (Gaussian kernel on standardised meta-features, landmarkers up-weighted).
  3. Predicted regret per model = similarity-weighted mean of its regrets, shrunk
     towards its global mean regret (empirical-Bayes style), so with weak
     neighbours we fall back to "what usually wins" instead of noise.
  4. Utility = -regret, adjusted by intent: interpretability and speed
     preferences from the parsed prompt.

`method="forest"` replaces step 2-3 with one random forest per model that regresses regret on the
meta-features, and `method="blend"` averages the two predictions.

Everything is evaluated with leave-one-dataset-out (see `evaluate`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DEFAULT_METRIC, INTERPRETABILITY, METRICS

PREF_WEIGHT = {"low": 0.0, "medium": 0.05, "high": 0.25}


@dataclass
class RankedModel:
    model: str
    utility: float
    predicted_regret: float
    expected_score: float | None
    interpretability: float
    relative_speed: float
    n_support: int
    confidence: float                          # share of the prediction driven by similar datasets vs the global prior
    evidence: list[tuple[str, float, float]]   # (neighbour dataset, weight, that model's score)

    def to_dict(self):
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


class MetaRanker:
    def __init__(self, kb: pd.DataFrame, bandwidth: float = 0.5, shrinkage: float = 1.0,
                 landmark_weight: float = 2.0, feature_set: str = "all", method: str = "kernel"):
        self.kb = kb
        self.method = method
        self.bandwidth = bandwidth
        self.shrinkage = shrinkage
        self.landmark_weight = landmark_weight
        self.feature_set = feature_set

    # ------------------------------------------------------------------
    def _tables(self, family: str, metric: str, exclude: str | None = None):
        kb = self.kb[self.kb.family == family]
        if exclude is not None:
            kb = kb[kb.dataset != exclude]
        col, higher = METRICS[family][metric]
        pcol = f"perf__{col}"
        if pcol not in kb or kb[pcol].notna().sum() == 0:
            col, higher = METRICS[family][DEFAULT_METRIC[family]]
            pcol = f"perf__{col}"
        perf = kb.pivot_table(index="dataset", columns="model", values=pcol, aggfunc="mean")
        sign = 1.0 if higher else -1.0
        best = (sign * perf).max(axis=1)
        worst = (sign * perf).min(axis=1)
        rng = (best - worst).replace(0, np.nan)
        regret = ((best.values[:, None] - sign * perf.values) / rng.values[:, None])
        regret = pd.DataFrame(regret, index=perf.index, columns=perf.columns).fillna(0.0).where(perf.notna())

        ttime = kb.pivot_table(index="dataset", columns="model", values="training_time", aggfunc="mean")
        mf_cols = [c for c in kb.columns if c.startswith("mf__")]
        meta = kb.groupby("dataset")[mf_cols].first().loc[perf.index]
        meta = meta.loc[:, meta.notna().mean() > 0.5]
        if self.feature_set == "landmarks":
            meta = meta[[c for c in meta.columns if "lm_" in c] or meta.columns]
        return perf, regret, ttime, meta, higher

    def _weights(self, meta: pd.DataFrame, query: dict) -> pd.Series:
        q = pd.Series({c: query.get(c.removeprefix("mf__"), np.nan) for c in meta.columns}, dtype=float)
        med = meta.median()
        M = meta.fillna(med)
        q = q.fillna(med)
        mu, sd = M.mean(), M.std(ddof=0).replace(0, 1).fillna(1)
        Z = (M - mu) / sd
        zq = ((q - mu) / sd).clip(-4, 4)
        Z = Z.clip(-4, 4)
        fw = np.array([self.landmark_weight if "lm_" in c else 1.0 for c in meta.columns])
        d2 = (((Z - zq) ** 2) * fw).sum(axis=1) / fw.sum()
        return np.exp(-d2 / (2 * self.bandwidth ** 2))

    def _forest(self, meta: pd.DataFrame, regret: pd.DataFrame, query: dict) -> dict[str, float]:
        from sklearn.ensemble import RandomForestRegressor
        med = meta.median()
        M = meta.fillna(med)
        q = pd.DataFrame([{c: query.get(c.removeprefix("mf__"), np.nan) for c in meta.columns}]).astype(float).fillna(med)
        out = {}
        for m in regret.columns:
            ok = regret[m].notna().to_numpy()
            if ok.sum() >= 3:
                rf = RandomForestRegressor(200, min_samples_leaf=3, random_state=0, n_jobs=1)
                out[m] = float(rf.fit(M[ok], regret[m][ok]).predict(q[M.columns])[0])
        return out

    def neighbours(self, family: str, meta_features: dict, metric: str | None = None,
                   exclude_dataset: str | None = None) -> list[tuple[str, float]]:
        metric = metric if metric in METRICS.get(family, {}) else DEFAULT_METRIC[family]
        _, _, _, meta, _ = self._tables(family, metric, exclude_dataset)
        w = self._weights(meta, meta_features).sort_values(ascending=False)
        return [(d, float(v)) for d, v in w.items()]

    # ------------------------------------------------------------------
    def rank(self, family: str, meta_features: dict, metric: str | None = None,
             interpretability: str = "medium", speed: str = "medium",
             exclude_dataset: str | None = None, candidates: list[str] | None = None) -> list[RankedModel]:
        metric = metric if metric in METRICS.get(family, {}) else DEFAULT_METRIC[family]
        perf, regret, ttime, meta, higher = self._tables(family, metric, exclude_dataset)
        w = self._weights(meta, meta_features)
        forest = self._forest(meta, regret, meta_features) if self.method in ("forest", "blend") else {}

        # speed: median log training time per model -> 0 (slowest) .. 1 (fastest)
        logt = np.log1p(ttime).median()
        speed_score = 1 - (logt - logt.min()) / ((logt.max() - logt.min()) or 1)

        out = []
        models = candidates or list(perf.columns)
        for m in models:
            if m not in regret.columns:
                continue
            r = regret[m]
            mask = r.notna()
            if mask.sum() == 0:
                continue
            wm = w[mask]
            prior = r[mask].mean()
            pred = (float((wm * r[mask]).sum()) + self.shrinkage * prior) / (float(wm.sum()) + self.shrinkage)
            if m in forest:
                pred = forest[m] if self.method == "forest" else 0.5 * (pred + forest[m])
            # expected absolute score: similarity-weighted mean of the model's raw scores
            p = perf[m][mask]
            exp_score = float((wm * p).sum() / wm.sum()) if wm.sum() > 1e-9 else float(p.mean())
            if metric in ("rmse", "mae", "mape", "smape"):
                exp_score = None   # scale-dependent: other datasets' units say nothing about this one
            interp = INTERPRETABILITY.get(m, 0.5)
            spd = float(speed_score.get(m, 0.5))
            util = -pred + PREF_WEIGHT[interpretability] * interp + PREF_WEIGHT[speed] * spd
            top = wm.sort_values(ascending=False).head(3)
            conf = float(wm.sum()) / (float(wm.sum()) + self.shrinkage)
            out.append(RankedModel(m, util, pred, exp_score, interp, spd, int(mask.sum()), conf,
                                   [(d, float(top[d]), float(perf.loc[d, m])) for d in top.index]))
        return sorted(out, key=lambda x: x.utility, reverse=True)


# ----------------------------------------------------------------------
# Leave-one-dataset-out evaluation
# ----------------------------------------------------------------------
def evaluate(kb: pd.DataFrame, family: str, metric: str | None = None, **ranker_kw) -> dict:
    metric = metric or DEFAULT_METRIC[family]
    full = MetaRanker(kb, **ranker_kw)
    perf, regret, _, meta, _ = full._tables(family, metric)
    res = {"top1_hit": [], "top2_hit": [], "regret_top1": [], "regret_best_of_top2": [],
           "baseline_top1_hit": [], "baseline_regret": [], "baseline_regret_best_of_top2": [], "random_regret": []}
    for ds in perf.index:
        r_true = regret.loc[ds].dropna()
        if len(r_true) < 2:
            continue
        mf = {c.removeprefix("mf__"): v for c, v in meta.loc[ds].items()}
        ranked = [x.model for x in MetaRanker(kb, **ranker_kw).rank(family, mf, metric, exclude_dataset=ds)
                  if x.model in r_true.index]
        best = r_true.idxmin()
        res["top1_hit"].append(ranked[0] == best)
        res["top2_hit"].append(best in ranked[:2])
        res["regret_top1"].append(r_true[ranked[0]])
        res["regret_best_of_top2"].append(r_true[ranked[:2]].min())
        # baseline: model with the lowest mean regret on the *other* datasets
        others = regret.drop(index=ds)[r_true.index].mean()
        base = others.idxmin()
        res["baseline_top1_hit"].append(base == best)
        res["baseline_regret"].append(r_true[base])
        res["baseline_regret_best_of_top2"].append(r_true[others.nsmallest(2).index].min())
        res["random_regret"].append(r_true.mean())
    return {k: float(np.mean(v)) for k, v in res.items()} | {"n_datasets": len(res["top1_hit"])}
