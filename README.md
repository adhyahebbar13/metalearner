# metalearner

Prompt-driven AutoML for tabular data. Describe the goal in plain English and point it at a CSV:

```bash
python -m metalearner.pipeline --csv Datasets/Classification/diabetes.csv \
  --prompt "Screen patients for diabetes; we must not miss positive cases" --trials 40
```

It works out the task (classification, regression, survival analysis, time-series forecasting or
anomaly detection), the target column and the metric that fits the goal (with Gemini, or a rule-based
parser without an API key), then uses a meta-learner trained on benchmarks of 205 datasets to pick the
models most likely to win on similar data, tunes the top two with Optuna and saves the fitted models
and a report.

* **Setup, usage, task types, leakage guards, benchmark protocol and ranker accuracy:**
  [metalearner/README.md](metalearner/README.md)
* **Data:** `Datasets/` holds the benchmark corpus (public datasets from Kaggle, OpenML, ADBench,
  Rdatasets and scikit-survival, each under its original licence); `metalearner/dataset_specs.py`
  records the task definition used for each file and which files are excluded and why.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-metalearner.txt     # macOS: brew install libomp (for XGBoost)
python -m pytest tests -q
```
