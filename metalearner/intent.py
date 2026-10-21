"""Prompt -> structured task specification.

Gemini (via REST, structured JSON output) is the primary parser. A deterministic
rule-based parser is used when no GEMINI_API_KEY is set or the call fails, and
every result - LLM or rules - is validated against the actual dataframe, so a
hallucinated column name can never reach the training code.

Replaces Code/llm_parser.py (Ollama phi3, hardcoded "BodyFat" fallback).
"""
from __future__ import annotations

import difflib
import json
import os
import re
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import requests

from .config import DEFAULT_METRIC, GEMINI_MODEL, METRICS, PARSER_TASKS, TASK_FAMILY


@dataclass
class TaskSpec:
    task: str                          # one of PARSER_TASKS
    target: str | None = None          # target column (event column for survival; known-anomaly labels or None)
    time_column: str | None = None     # survival duration, or time-series timestamp
    event_column: str | None = None    # survival event indicator
    exclude_columns: list[str] = field(default_factory=list)
    primary_metric: str | None = None
    interpretability: str = "medium"   # low | medium | high
    speed: str = "medium"              # low | medium | high (how much speed matters)
    positive_class: str | None = None
    forecast_horizon: int | None = None
    expected_anomaly_rate: float | None = None   # share of rows to flag (anomaly detection)
    application_summary: str = ""
    source: str = "rules"              # gemini | rules
    warnings: list[str] = field(default_factory=list)

    @property
    def family(self) -> str:
        return TASK_FAMILY[self.task]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["family"] = self.family
        return d


# ----------------------------------------------------------------------------
# Gemini
# ----------------------------------------------------------------------------
_ALL_METRICS = sorted({m for fam in METRICS.values() for m in fam})

_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "task": {"type": "STRING", "enum": PARSER_TASKS},
        "target": {"type": "STRING", "nullable": True,
                   "description": "Exact column name to predict. For survival, the event/status column. "
                                  "For anomaly detection, a column of known anomaly labels, or null if none."},
        "time_column": {"type": "STRING", "nullable": True,
                        "description": "Survival duration column, or the timestamp column for forecasting."},
        "event_column": {"type": "STRING", "nullable": True},
        "exclude_columns": {"type": "ARRAY", "items": {"type": "STRING"},
                            "description": "Identifier or leakage columns that must not be used as features."},
        "primary_metric": {"type": "STRING", "enum": _ALL_METRICS},
        "interpretability": {"type": "STRING", "enum": ["low", "medium", "high"]},
        "speed": {"type": "STRING", "enum": ["low", "medium", "high"]},
        "positive_class": {"type": "STRING", "nullable": True},
        "forecast_horizon": {"type": "INTEGER", "nullable": True},
        "expected_anomaly_rate": {"type": "NUMBER", "nullable": True,
                                  "description": "Fraction (0-1) of rows the user expects to be anomalous, if stated."},
        "application_summary": {"type": "STRING"},
    },
    "required": ["task", "target", "primary_metric", "interpretability", "speed",
                 "exclude_columns", "application_summary"],
}

_SYSTEM = """You turn a user's description of a machine-learning goal plus a dataset
schema into a task specification. Rules:
- Use column names EXACTLY as listed in the schema.
- Pick primary_metric from the user's application: screening / "don't miss cases" -> recall;
  "avoid false alarms" -> precision; ranking risk -> roc_auc; balanced or unspecified
  classification -> f1; regression in natural units -> rmse or mae; survival -> c_index;
  forecasting -> rmse; anomaly detection -> pr_auc (default), roc_auc for ranking/"don't miss",
  precision_at_n for "avoid false alarms".
- anomaly_detection: only set target if a column already labels known anomalies; otherwise null.
- interpretability=high if the user needs to explain decisions (clinicians, regulators, "why").
- speed=high if the user needs fast training/inference (real-time, edge, quick).
- exclude_columns: IDs, names, free-text notes and anything that leaks the target.
- For survival_analysis set time_column (duration) and event_column (status); target = event_column.
"""


def _schema_summary(df: pd.DataFrame, max_cols: int = 80) -> str:
    lines = []
    for col in df.columns[:max_cols]:
        s = df[col]
        sample = ", ".join(map(str, s.dropna().unique()[:4]))
        lines.append(f"- {col!r}: dtype={s.dtype}, n_unique={s.nunique()}, "
                     f"missing={s.isna().mean():.0%}, e.g. [{sample}]")
    if len(df.columns) > max_cols:
        lines.append(f"... and {len(df.columns) - max_cols} more columns")
    return f"rows={len(df)}\n" + "\n".join(lines)


def parse_with_gemini(prompt: str, df: pd.DataFrame, api_key: str,
                      model: str = GEMINI_MODEL, timeout: int = 60) -> dict:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {
        "systemInstruction": {"parts": [{"text": _SYSTEM}]},
        "contents": [{"role": "user", "parts": [{"text":
            f"User goal:\n{prompt}\n\nDataset schema:\n{_schema_summary(df)}"}]}],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
            "responseSchema": _RESPONSE_SCHEMA,
        },
    }
    # key in a header, not the URL, so it never appears in error messages or logs
    r = requests.post(url, headers={"x-goog-api-key": api_key}, json=body, timeout=timeout)
    if not r.ok:
        try:
            detail = r.json()["error"]["message"]
        except Exception:
            detail = r.text[:300]
        raise RuntimeError(f"{r.status_code} from {model}: {detail}")
    text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text)


# ----------------------------------------------------------------------------
# Rule-based fallback
# ----------------------------------------------------------------------------
_TASK_RULES = [
    ("survival_analysis", r"surviv|time[- ]to[- ](event|death|relapse|failure|recurrence|progression)|hazard|censor"
                          r"|until (death|relapse|recurrence)|how long (until|before|will)|mortality over time|prognos"),
    ("time_series_forecasting", r"forecast|next (day|week|month|hour|reading)|future values|over time|time[- ]series|trend"),
    ("anomaly_detection", r"anomal|outlier|unusual|abnormal reading|fraud|novelty|suspicious|odd (one|record|reading)s?"),
    ("regression", r"how much|how many|estimate|predict (the )?(level|value|amount|score|cost|charges|price)|continuous|regress"),
    ("multiclass_classification", r"categor|which (type|class|stage|level)|multi-?class|classes|stage"),
    ("binary_classification", r"whether|if (a|the)? ?patient|yes/no|binary|has |have |detect|diagnos|risk of|classif"),
]
_METRIC_RULES = {
    "classification": [("recall", r"miss|sensitiv|recall|screen|catch (all|every)|false negative"),
                       ("precision", r"false (alarm|positive)|precision|only flag"),
                       ("roc_auc", r"\bauc\b|rank|risk score|probabilit"),
                       ("accuracy", r"accura")],
    "regression": [("mae", r"absolute|mae|typical error"), ("r2", r"r2|r\^2|variance explained|r-squared")],
    "timeseries": [("mae", r"absolute|mae"), ("smape", r"percent|mape|smape")],
    "survival": [],
    "anomaly": [("precision_at_n", r"false (alarm|positive)|precision|only flag|top \d+"),
                ("roc_auc", r"miss|sensitiv|rank|\bauc\b|roc"),
                ("pr_auc", r"pr[- ]?auc|average precision")],
}
_ANOMALY_LABEL_PAT = re.compile(r"^(is_?)?(anomal\w*|outlier|label|fraud|is_fraud|class|target)$", re.I)
_SURV_TIME_KW = ["time", "duration", "survival", "followup", "follow_up", "months", "days", "years", "futime"]
_SURV_EVENT_KW = ["event", "status", "censor", "death", "deceased", "vital", "fustat", "dead", "died"]
# exact (normalised) names that are almost always the survival time / event column
_SURV_TIME_EXACT = {"time", "futime", "survival_time", "os_time", "os_months", "overall_survival_months",
                    "follow_up", "followup", "duration", "time_to_event", "days_to_event", "survival_months"}
_SURV_EVENT_EXACT = {"status", "event", "death", "dead", "died", "fustat", "fstat", "vital_status", "os_status",
                     "overall_survival", "death_event", "censor", "censored", "censoring", "delta"}
_ID_PAT = re.compile(r"((^|_)(id|patient_?id|record_?id)|^index|^unnamed.*)$", re.I)
_ID_NAME_PAT = re.compile(r"(^|[^a-z])(id|ids|patient|patient_?no|subject|record|encounter_id|mrn)([^a-z]|$)", re.I)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _snake(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def _is_sequential_id(s: pd.Series) -> bool:
    """A column of distinct consecutive integers (row numbers / patient numbers)."""
    v = pd.to_numeric(s, errors="coerce")
    if len(v) < 20 or v.isna().any() or (v % 1 != 0).any() or v.nunique() != len(v):
        return False
    return bool((np.diff(np.sort(v.to_numpy())) == 1).all())


def id_like_columns(df: pd.DataFrame, keep=()) -> list[str]:
    """Identifier columns: ID-ish names that are (nearly) unique per row, or sequential integers."""
    out = []
    for c in df.columns:
        if c in keep:
            continue
        s = df[c]
        unique_ratio = s.nunique() / max(len(s), 1)
        if (_ID_PAT.search(str(c)) and unique_ratio >= 0.3) or (_ID_NAME_PAT.search(str(c)) and unique_ratio > 0.5) \
                or _is_sequential_id(s):
            out.append(c)
    return out


def _kw_score(name: str, exact: set, keywords: list) -> int:
    n = _snake(name)
    if n in exact:
        return 3
    if any(n.startswith(k) or n.endswith(k) for k in keywords):
        return 2
    return 1 if any(k in n for k in keywords) else 0


def find_survival_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Keyword + value-shape heuristic. The event must have 2-3 distinct values and the time must be a
    non-negative number with many distinct values; exact names beat partial matches, so e.g.
    'overall_survival' wins over 'er_status_measured_by_ihc' and 'futime' over 'pctime'."""
    times, events = [], []
    for col in df.columns:
        s = df[col]
        name = _snake(col)
        if "patient" in name or "id" in name.split("_") or "treatment" in name:
            continue
        nun = s.nunique()
        miss = float(s.isna().mean())
        if pd.api.types.is_numeric_dtype(s) and nun >= 4 and (s.dropna() >= 0).all():
            sc = _kw_score(col, _SURV_TIME_EXACT, _SURV_TIME_KW)
            if sc:
                times.append((sc, -miss, nun, col))
        if 2 <= nun <= 3:
            sc = _kw_score(col, _SURV_EVENT_EXACT, _SURV_EVENT_KW)
            if sc:
                events.append((sc, -miss, -nun, col))
    time_col = max(times)[3] if times else None
    event_col = max(events)[3] if events else None
    if time_col is None or event_col is None:
        if len(df.columns) >= 2:
            time_col, event_col = time_col or df.columns[-2], event_col or df.columns[-1]
    return time_col, event_col


def _mentioned_column(prompt: str, columns) -> str | None:
    p = _norm(prompt)
    hits = [c for c in columns if len(_norm(c)) >= 3 and _norm(c) in p]
    if hits:
        return max(hits, key=lambda c: len(_norm(c)))  # longest = most specific
    # fuzzy on words of the prompt
    words = re.findall(r"[A-Za-z_]{4,}", prompt)
    best, best_score = None, 0.0
    for w in words:
        for c in columns:
            sc = difflib.SequenceMatcher(None, w.lower(), str(c).lower()).ratio()
            if sc > best_score:
                best, best_score = c, sc
    return best if best_score >= 0.8 else None


def parse_with_rules(prompt: str, df: pd.DataFrame) -> dict:
    p = prompt.lower()
    task = next((t for t, pat in _TASK_RULES if re.search(pat, p)), None)
    target = _mentioned_column(prompt, df.columns)
    out: dict = {"exclude_columns": [], "application_summary": prompt.strip()[:300]}
    if task is None:
        # infer from the target if we have one
        task = _infer_task_from_target(df[target]) if target else "binary_classification"
    out["task"] = task
    fam = TASK_FAMILY[task]
    if fam == "survival":
        t, e = find_survival_columns(df)
        out.update(time_column=t, event_column=e, target=e)
    elif fam == "anomaly":   # labels are optional: only an explicitly named or label-like column
        out["target"] = target or next((c for c in df.columns if _ANOMALY_LABEL_PAT.match(str(c))), None)
        m = re.search(r"(\d+(?:\.\d+)?)\s*%", p)
        out["expected_anomaly_rate"] = float(m.group(1)) / 100 if m else None
    else:
        out["target"] = target or df.columns[-1]   # repo benchmark convention: last column
    out["primary_metric"] = next((m for m, pat in _METRIC_RULES.get(fam, []) if re.search(pat, p)), None)
    out["interpretability"] = "high" if re.search(r"explain|interpret|transparen|clinician|doctor|why|regulat", p) else "medium"
    out["speed"] = "high" if re.search(r"fast|real[- ]time|quick|lightweight|edge device|low latency", p) else "medium"
    m = re.search(r"next (\d+)", p)
    out["forecast_horizon"] = int(m.group(1)) if m else None
    return out


# ----------------------------------------------------------------------------
# Validation against the data
# ----------------------------------------------------------------------------
def _resolve_column(name, columns) -> str | None:
    if name is None:
        return None
    if name in columns:
        return name
    lut = {_norm(c): c for c in columns}
    if _norm(name) in lut:
        return lut[_norm(name)]
    m = difflib.get_close_matches(str(name), [str(c) for c in columns], n=1, cutoff=0.75)
    return m[0] if m else None


def _infer_task_from_target(y: pd.Series) -> str:
    y = y.dropna()
    n_unique = y.nunique()
    if n_unique <= 2:
        return "binary_classification"
    if y.dtype == object or n_unique <= min(20, max(3, int(0.05 * len(y)))):
        return "multiclass_classification"
    return "regression"


def validate(raw: dict, df: pd.DataFrame, source: str) -> TaskSpec:
    cols = list(df.columns)
    warns: list[str] = []
    task = raw.get("task") if raw.get("task") in PARSER_TASKS else "binary_classification"

    spec = TaskSpec(task=task, source=source,
                    interpretability=raw.get("interpretability") or "medium",
                    speed=raw.get("speed") or "medium",
                    positive_class=raw.get("positive_class"),
                    forecast_horizon=raw.get("forecast_horizon"),
                    application_summary=raw.get("application_summary", ""))

    if spec.family == "survival":
        t = _resolve_column(raw.get("time_column"), cols)
        e = _resolve_column(raw.get("event_column") or raw.get("target"), cols)
        if t is None or e is None:
            ht, he = find_survival_columns(df)
            t, e = t or ht, e or he
            warns.append(f"Survival columns partly inferred by heuristic: time={t!r}, event={e!r}")
        spec.time_column, spec.event_column, spec.target = t, e, e
    elif spec.family == "anomaly":
        target = _resolve_column(raw.get("target"), cols)
        if raw.get("target") and target is None:
            warns.append(f"Label column {raw.get('target')!r} not found; running unsupervised.")
        if target is not None and df[target].nunique() > 20 and pd.api.types.is_numeric_dtype(df[target]):
            warns.append(f"{target!r} looks continuous, not like anomaly labels; using it as a feature instead.")
            target = None
        if target is None:
            warns.append("No anomaly labels: models are ranked by the meta-learner but cannot be tuned or scored.")
        spec.target = target
        rate = raw.get("expected_anomaly_rate")
        spec.expected_anomaly_rate = float(rate) if isinstance(rate, (int, float)) and 0 < rate < 0.5 else None
    else:
        target = _resolve_column(raw.get("target"), cols)
        if target is None:
            target = cols[-1]
            warns.append(f"Target {raw.get('target')!r} not found in columns; using last column {target!r}.")
        elif target != raw.get("target"):
            warns.append(f"Target {raw.get('target')!r} matched to column {target!r}.")
        spec.target = target
        if spec.family == "timeseries":
            spec.time_column = _resolve_column(raw.get("time_column"), cols)

        # Consistency check between stated task and the actual target column
        if spec.family in ("classification", "regression"):
            inferred = _infer_task_from_target(df[target])
            if spec.family == "classification" and inferred == "regression":
                warns.append(f"Target {target!r} looks continuous ({df[target].nunique()} unique values) "
                             f"but the prompt asks for classification; keeping classification.")
            elif spec.family == "regression" and inferred != "regression":
                warns.append(f"Target {target!r} has only {df[target].nunique()} unique values; "
                             f"switching regression -> {inferred}.")
                spec.task = inferred
            elif spec.family == "classification" and inferred != spec.task:
                spec.task = inferred   # binary vs multiclass is decided by the data

    spec.exclude_columns = [c for c in (_resolve_column(x, cols) for x in raw.get("exclude_columns") or [])
                            if c and c not in (spec.target, spec.time_column, spec.event_column)]
    for c in id_like_columns(df, keep=(spec.target, spec.time_column, spec.event_column)):
        if c not in spec.exclude_columns:
            spec.exclude_columns.append(c)

    metric = raw.get("primary_metric")
    fam_metrics = METRICS.get(spec.family, METRICS["classification"])
    if metric not in fam_metrics:
        if metric:
            warns.append(f"Metric {metric!r} not valid for {spec.family}; using default.")
        metric = DEFAULT_METRIC.get(spec.family, "f1")
    spec.primary_metric = metric
    spec.warnings = warns
    return spec


def parse_intent(prompt: str, df: pd.DataFrame, api_key: str | None = None, verbose: bool = True) -> TaskSpec:
    api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    raw, source = None, "rules"
    if api_key:
        try:
            raw, source = parse_with_gemini(prompt, df, api_key), "gemini"
        except Exception as exc:  # network, quota, bad JSON ...
            if verbose:
                print(f"[intent] Gemini failed ({exc.__class__.__name__}: {exc}); using rule-based parser.")
    if raw is None:
        raw = parse_with_rules(prompt, df)
    return validate(raw, df, source)
