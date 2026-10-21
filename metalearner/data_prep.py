"""Turn a raw CSV + TaskSpec into model-ready X / y, without leakage.

All learned preprocessing (imputation, scaling, one-hot) lives inside an sklearn
Pipeline built by `make_preprocessor`, so it is re-fitted inside every CV fold.
(The original benchmark scripts scaled the full dataset before CV.)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from .anomaly import binarize_labels
from .intent import TaskSpec


def load_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]
    df = df.dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]
    return df


@dataclass
class Prepared:
    X: pd.DataFrame
    y: object                     # np.ndarray, or sksurv-style structured array for survival
    family: str
    numeric_cols: list[str]
    categorical_cols: list[str]
    classes_: list | None = None  # label order for classification
    notes: list[str] = field(default_factory=list)
    row_ids: list | None = None   # original CSV row of each X row (anomaly detection output)


def _split_feature_types(X: pd.DataFrame, notes: list[str]) -> tuple[list[str], list[str], pd.DataFrame]:
    num, cat, drop = [], [], []
    for c in X.columns:
        s = X[c]
        if pd.api.types.is_bool_dtype(s):
            X[c] = s.astype(float)
            num.append(c)
        elif pd.api.types.is_numeric_dtype(s):
            num.append(c)
        else:
            coerced = pd.to_numeric(s, errors="coerce")
            if coerced.notna().mean() > 0.95 and s.notna().sum() > 0:  # numbers stored as strings
                X[c] = coerced
                num.append(c)
            elif s.nunique() > max(50, 0.5 * len(s)):                   # free text / identifiers
                drop.append(c)
            else:
                X[c] = s.astype(str).where(s.notna(), None)
                cat.append(c)
    if drop:
        notes.append(f"Dropped high-cardinality text columns: {drop}")
        X = X.drop(columns=drop)
    return num, cat, X


# |Spearman| against a continuous outcome / single-feature AUC against a binary one.
# Measured on the 120 benchmark datasets: real predictors reach |rho| < 0.97 and AUC <= 0.976
# (breast-cancer cell size), while body fat vs the density it is computed from is 0.993.
LEAK_RHO, LEAK_AUC = 0.99, 0.995
SUSPECT_RHO, SUSPECT_AUC = 0.97, 0.98


def _leaky_columns(X: pd.DataFrame, outcomes: dict[str, pd.Series],
                   suspects: list | None = None) -> list[tuple[str, str]]:
    """Features that (almost) reproduce an outcome: exact re-codings of a discrete outcome, or
    |Spearman| / AUC above LEAK_RHO / LEAK_AUC. Copies, derived values (e.g. density -> body fat)
    and IDs sorted by time get that close; real predictors do not. Near-misses go to `suspects`."""
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    found = []
    suspects = suspects if suspects is not None else []
    for c in X.columns:
        x = X[c]
        for name, y in outcomes.items():
            ok = x.notna().to_numpy() & y.notna().to_numpy()
            if ok.sum() < 10:
                continue
            xv, yv = x[ok], y[ok]
            discrete_y = yv.nunique() <= 10
            small_ok = ok.sum() >= 30 or xv.nunique() == yv.nunique()   # tiny data: only 1-to-1 recodings
            if discrete_y and small_ok and 1 < xv.nunique() <= 20 and xv.nunique() < ok.sum() / 2 \
                    and (pd.DataFrame({"x": xv.astype(str).values, "y": yv.values}).groupby("x").y.nunique() == 1).all():
                found.append((c, f"exactly determines {name}"))
                break
            if ok.sum() < 30 or not pd.api.types.is_numeric_dtype(xv) or xv.nunique() < 3:
                continue
            if discrete_y and yv.nunique() == 2:
                auc = roc_auc_score((yv == yv.max()).astype(int), xv)
                auc = max(auc, 1 - auc)
                if auc >= LEAK_AUC:
                    found.append((c, f"AUC {auc:.3f} against {name}"))
                    break
                if auc >= SUSPECT_AUC:
                    suspects.append((c, f"AUC {auc:.3f} against {name}"))
            elif not discrete_y and pd.api.types.is_numeric_dtype(yv):
                r = abs(spearmanr(xv, yv)[0])
                if np.isfinite(r) and r >= LEAK_RHO:
                    found.append((c, f"|Spearman| {r:.3f} with {name}"))
                    break
                if np.isfinite(r) and r >= SUSPECT_RHO:
                    suspects.append((c, f"|Spearman| {r:.3f} with {name}"))
    return found


def _drop_leaks(X: pd.DataFrame, outcomes: dict, notes: list[str]) -> pd.DataFrame:
    suspects = []
    leaks = _leaky_columns(X, outcomes, suspects)
    if suspects:
        notes.append("Kept, but check these do not leak the outcome: "
                     + "; ".join(f"{c!r} ({why})" for c, why in suspects))
    if leaks:
        notes.append("Dropped columns that leak the outcome: " + "; ".join(f"{c!r} ({why})" for c, why in leaks))
        X = X.drop(columns=[c for c, _ in leaks])
    return X


def prepare(df: pd.DataFrame, spec: TaskSpec) -> Prepared:
    notes: list[str] = []
    df = df.copy()
    fam = spec.family
    drop_cols = set(spec.exclude_columns)

    if fam == "anomaly":
        # detectors never see labels; a label column (if any) is only used to score them
        y = None
        if spec.target and spec.target in df.columns:
            df = df[df[spec.target].notna()]
            y, rule = binarize_labels(df[spec.target], spec.positive_class)
            notes.append(f"Labels from {spec.target!r}: {rule} ({int(y.sum())} of {len(y)} rows).")
            if y.sum() < 2:
                notes.append("Fewer than 2 labelled anomalies: running unsupervised (no tuning).")
                y = None
        X = df.drop(columns=[c for c in [spec.target, *drop_cols] if c and c in df.columns])
        num, cat, X = _split_feature_types(X, notes)
        return Prepared(X.reset_index(drop=True), y, fam, num, cat, ["normal", "anomaly"], notes,
                        row_ids=X.index.tolist())

    if fam == "survival":
        t, e = spec.time_column, spec.event_column
        df[t] = pd.to_numeric(df[t], errors="coerce")
        ev = df[e]
        if spec.positive_class is not None:   # explicit "this value means the event happened"
            pos = str(spec.positive_class).strip()
            num_pos = pd.to_numeric(pd.Series([pos]), errors="coerce")[0]
            is_pos = ev.astype(str).str.strip().eq(pos) | (pd.to_numeric(ev, errors="coerce") == num_pos)
            ev = is_pos.astype(float).where(ev.notna())
            notes.append(f"Event = {e!r} == {pos!r}.")
        if not pd.api.types.is_numeric_dtype(ev):
            ev = pd.Series(LabelEncoder().fit_transform(ev.astype(str)), index=df.index)
            notes.append(f"Event column {e!r} label-encoded; check that 1 = event occurred.")
        df[e] = pd.to_numeric(ev, errors="coerce")
        df = df[df[t].notna() & df[e].notna() & (df[t] > 0)]
        events = df[e].astype(float)
        if spec.positive_class is not None:
            pass
        elif events.nunique() > 2:   # e.g. 1/2 coding (R survival: 1=censored, 2=dead)
            events = (events == events.max()).astype(float)
            notes.append(f"Event column {e!r} had >2 codes; treating max value as the event.")
        elif set(events.unique()) <= {1.0, 2.0}:
            events = (events == 2).astype(float)
            notes.append(f"Event column {e!r} coded 1/2; treating 2 as the event.")
        y = np.array(list(zip(events.astype(bool), df[t].astype(float))),
                     dtype=[("event", bool), ("time", float)])
        X = df.drop(columns=[c for c in [t, e, *drop_cols] if c in df.columns])
        num, cat, X = _split_feature_types(X, notes)
        X = _drop_leaks(X, {"the event": pd.Series(events.values, index=X.index),
                            "the survival time": df[t]}, notes)
        num, cat = [c for c in num if c in X], [c for c in cat if c in X]
        return Prepared(X.reset_index(drop=True), y, fam, num, cat, notes=notes)

    target = spec.target
    df = df[df[target].notna()]

    if fam == "timeseries":
        if spec.time_column and spec.time_column in df.columns:
            ts = pd.to_datetime(df[spec.time_column], errors="coerce")
            if ts.notna().mean() > 0.9:
                df = df.assign(_ts=ts).sort_values("_ts").drop(columns="_ts")
            drop_cols.add(spec.time_column)
        y = pd.to_numeric(df[target], errors="coerce")
        keep = y.notna()
        df, y = df[keep], y[keep].to_numpy(dtype=float)
        X = df.drop(columns=[c for c in [target, *drop_cols] if c in df.columns])
        X = X.select_dtypes(include=[np.number])  # forecasting models use numeric exogenous only
        return Prepared(X.reset_index(drop=True), y, fam, list(X.columns), [], notes=notes)

    X = df.drop(columns=[c for c in [target, *drop_cols] if c in df.columns])
    num, cat, X = _split_feature_types(X, notes)
    y_check = df[target].astype(str).str.strip() if fam == "classification" else pd.to_numeric(df[target], errors="coerce")
    if fam == "classification" and y_check.nunique() == 2:
        y_check = pd.Series(pd.factorize(y_check)[0], index=y_check.index)
    X = _drop_leaks(X, {f"the target {target!r}": y_check}, notes)
    num, cat = [c for c in num if c in X], [c for c in cat if c in X]
    classes = None
    if fam == "classification":
        y_raw = df[target].astype(str).str.strip()
        counts = y_raw.value_counts()
        rare = counts[counts < 5].index.tolist()
        if rare:
            notes.append(f"Dropped rows of very rare classes {rare} (<5 samples each).")
            mask = ~y_raw.isin(rare)
            X, y_raw = X[mask.values], y_raw[mask]
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        classes = le.classes_.tolist()
    else:
        y = pd.to_numeric(df[target], errors="coerce")
        mask = y.notna().values
        X, y = X[mask], y[mask].to_numpy(dtype=float)
    return Prepared(X.reset_index(drop=True), y, fam, num, cat, classes, notes)


def make_preprocessor(prep: Prepared, scale: bool = True) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    transformers = []
    if prep.numeric_cols:
        transformers.append(("num", Pipeline(num_steps), prep.numeric_cols))
    if prep.categorical_cols:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=20,
                                     sparse_output=False)),
        ]), prep.categorical_cols))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def simple_numeric_matrix(prep: Prepared) -> np.ndarray:
    """Quick imputed+scaled+encoded matrix, used only for meta-feature computation."""
    return make_preprocessor(prep).fit_transform(prep.X)
