"""Run: python -m pytest tests -q   (or: python tests/test_intent.py)"""
import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metalearner.intent import parse_intent  # noqa: E402

DF = pd.DataFrame({
    "patient_id": range(200),
    "Glucose": np.random.default_rng(0).normal(120, 30, 200),
    "BMI": np.random.default_rng(1).normal(30, 5, 200),
    "Outcome": np.random.default_rng(2).integers(0, 2, 200),
})


def _fake_gemini(payload: dict):
    resp = mock.Mock()
    resp.raise_for_status = lambda: None
    resp.json = lambda: {"candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]}
    return resp


def test_gemini_happy_path_and_column_repair():
    payload = {"task": "binary_classification", "target": "outcome",   # wrong case -> repaired
               "primary_metric": "recall", "interpretability": "high", "speed": "low",
               "exclude_columns": ["Patient ID"], "application_summary": "diabetes screening"}
    with mock.patch("metalearner.intent.requests.post", return_value=_fake_gemini(payload)) as post:
        spec = parse_intent("screen for diabetes", DF, api_key="k", verbose=False)
    body = post.call_args.kwargs["json"]
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert "Outcome" in body["contents"][0]["parts"][0]["text"]          # schema was sent
    assert spec.source == "gemini" and spec.target == "Outcome"
    assert spec.primary_metric == "recall" and spec.interpretability == "high"
    assert "patient_id" in spec.exclude_columns


def test_gemini_hallucinated_target_falls_back():
    payload = {"task": "regression", "target": "HbA1c", "primary_metric": "f1",
               "interpretability": "medium", "speed": "medium", "exclude_columns": [], "application_summary": ""}
    with mock.patch("metalearner.intent.requests.post", return_value=_fake_gemini(payload)):
        spec = parse_intent("predict hba1c", DF, api_key="k", verbose=False)
    assert spec.target == "Outcome"                        # last column fallback
    assert spec.task == "binary_classification"            # 2-valued target overrides 'regression'
    assert spec.primary_metric == "f1"
    assert any("not found" in w for w in spec.warnings)


def test_gemini_failure_uses_rules():
    with mock.patch("metalearner.intent.requests.post", side_effect=TimeoutError("down")):
        spec = parse_intent("predict whether the patient has diabetes (Outcome), don't miss cases", DF,
                            api_key="k", verbose=False)
    assert spec.source == "rules" and spec.target == "Outcome" and spec.primary_metric == "recall"


def test_rules_survival():
    df = pd.DataFrame({"time": [5, 10, 15, 20] * 10, "status": [1, 0, 1, 1] * 10, "age": range(40)})
    spec = parse_intent("how long until death for these patients", df, api_key=None, verbose=False)
    assert spec.family == "survival" and spec.time_column == "time" and spec.event_column == "status"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
