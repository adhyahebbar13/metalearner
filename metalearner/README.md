# metalearner — prompt-driven model selection + tuning

```
 user prompt ──► intent.py (Gemini, structured JSON; rule fallback; validated against the CSV)
                    │  task, target, survival/time cols, metric, interpretability, speed, anomaly rate
 dataset CSV ──► data_prep.py (ID + leak checks) ──► meta_features.py (stats + info-theory + landmarkers)
                                         │
 artifacts/benchmarks/*_detailed.csv ──► knowledge_base.py ──► ranker.py (relative performance of each
 (benchmark.py over Datasets/*)                                 model on similar datasets, intent-adjusted)
                                                                 │ ranked models
                                                                 ▼
                                    hpo.py: CASH over {top-1, top-2} — Optuna TPE, warm-started with the
                                    benchmark best_params of the most similar datasets, fold pruning
                                    (anomaly detection without labels: no tuning, neighbours' params)
                                                                 │
                                                                 ▼
                                    artifacts/runs/<time>_<dataset>/report.json + fitted .joblib models
                                    (+ anomaly_scores.csv for anomaly detection)
```

## Run

```bash
pip install -r requirements-metalearner.txt      # macOS + XGBoost also needs: brew install libomp
export GEMINI_API_KEY=...                        # optional; without it the rule-based parser is used
export GEMINI_MODEL=gemini-3.5-flash             # optional; default gemini-3.8-flash

python -m metalearner.pipeline \
  --csv "Datasets/Classification/diabetes.csv" \
  --prompt "Screen patients for diabetes; we must not miss positive cases and doctors must understand it" \
  --trials 40
```

Flags: `--top-k 2` models to tune, `--timeout` seconds for HPO, `--backend auto|optuna|random`,
`--exclude-self` removes the dataset's own benchmark rows from the KB (use this for honest demos on repo datasets).

### Task types

| task | models | metrics (default first) |
|---|---|---|
| binary / multiclass classification | LogisticRegression, RandomForest, GradientBoosting, XGBoost, CatBoost, SVC, KNN | f1, accuracy, precision, recall, roc_auc |
| regression | LinearRegression, Ridge, Lasso, RandomForest, GradientBoosting, XGBoost, CatBoost, SVR, KNN | rmse, mae, r2, mape |
| survival analysis | CoxPH, RandomSurvivalForest, GradientBoostingSurvival | c_index |
| time-series forecasting | lag-XGBoost, ARIMA, SARIMA, Prophet | rmse, mae, smape, r2 |
| anomaly detection | IsolationForest, LOF, OneClassSVM, KNN, ECOD, HBOS, Mahalanobis | pr_auc, roc_auc, precision_at_n |

Asking for recall/precision in classification optimises F2/F0.5 (pure recall is maxed by predicting
everything positive) and reports both.

**Anomaly detection** works with or without labels. Detectors are always fitted unsupervised; a label
column (e.g. `is_anomaly`, or any column the prompt names) is only used to tune and score them. Without
labels the meta-learner's top pick is fitted with the hyper-parameters that won on the most similar
benchmark dataset. Either way `anomaly_scores.csv` holds an out-of-fold score, rank and flag for every
row; the flag rate is the label rate, the rate stated in the prompt ("about 5% are faulty"), or 5%.

### Leakage guards (data_prep.py / intent.py)

* identifier columns are dropped: ID-like names that are unique per row, and any column of distinct
  consecutive integers (row / patient numbers, even when the file is sorted by outcome);
* features that reproduce the outcome are dropped with a note: exact re-codings of a discrete target
  or event (e.g. `Dead or Alive` next to `Event`), or |Spearman| / AUC ≥ 0.995 against the target,
  the event or the survival time;
* all fitted preprocessing lives inside the CV pipeline.

## Knowledge base

The KB is built from benchmarks re-run with the pipeline's own CV + Optuna search, so every score is
measured the same leakage-free way:

```bash
python -m metalearner.benchmark --family anomaly --jobs 4     # per family; resumable (artifacts/benchmarks/_parts)
python -m metalearner.knowledge_base                          # join meta-features with benchmark scores
python scripts/evaluate_metalearner.py                        # leave-one-dataset-out accuracy of the ranker
```

`--jobs N` runs N datasets in parallel and splits the cores between them (`METALEARNER_N_JOBS`); never
run two benchmarks at once — OpenMP libraries slow down drastically when oversubscribed.

Benchmark protocol: every model gets its library default plus 15 Optuna trials or 5 minutes, whichever
comes first (the same budget a user's run faces), tuned on the family's default metric with 5-fold CV
(3-fold rolling origin for time series); the other metrics are then measured for the winning config.
Large datasets are capped at 5,000 rows (stratified). Tuning and reporting use the same folds, so
absolute scores are slightly optimistic — equally for every model, which is what the ranking needs.

### Current knowledge base and ranker accuracy

205 datasets, 1,247 (dataset, model) rows. Leave-one-dataset-out (`scripts/evaluate_metalearner.py`);
regret = how far the chosen model is from the best one on that dataset (0 = best, 1 = worst).
"Baseline" always picks the models with the lowest average regret on the other datasets.

| task | datasets | ranker top-1 regret | baseline top-1 | ranker best-of-top-2 | baseline best-of-top-2 | random |
|---|---|---|---|---|---|---|
| classification | 60 | **0.221** | 0.323 | 0.107 | 0.109 | 0.399 |
| regression | 33 | 0.100 | **0.069** | 0.043 | 0.043 | 0.393 |
| survival | 29 | 0.634 | 0.735 | 0.248 | **0.137** | 0.460 |
| time series | 46 | **0.124** | 0.201 | **0.033** | 0.056 | 0.418 |
| anomaly | 37 | 0.388 | **0.314** | **0.181** | 0.288 | 0.473 |

The pipeline tunes the top 2, so the best-of-top-2 columns are the ones that matter most. Survival has only
three models whose C-indices are usually within noise of each other, so no ranking does much better than
chance there. `MetaRanker(method="forest" | "blend")` are alternatives (per-model random forests on the
meta-features); choosing the method or its bandwidth/shrinkage per task by nested LODO did not beat the
fixed default, so the default stays `kernel`.

`dataset_specs.py` holds the hand-checked task for each benchmark dataset (target, survival time/event and
event value, leaking columns, row filters) and the datasets left out with the reason. The original
`Benchmark_results/` used the last column as the target, which was wrong for about half the
classification/regression files (e.g. `Wrist` instead of `BodyFat`), and several survival and time-series
files were not survival / time-series data at all; those files are only used for a family that has not
been re-benchmarked.

Datasets added for the KB (re-download with the scripts):

* `scripts/fetch_kb_datasets.py` — OpenML classification (`openml_*`) and regression (OpenML-CTR23),
  survival data from Rdatasets / scikit-survival (`r_*`, `sksurv_*`), time series from Rdatasets (`ts_*`)
  and series extracted from the repo's own files (`derived_*`);
* `scripts/make_anomaly_datasets.py` — `Datasets/Anomaly/`: ADBench "Classical" datasets plus three
  medical classification sets turned into anomaly sets (one class normal, the other downsampled).

## Tests

```bash
python tests/test_intent.py    # parser (Gemini mocked)
python tests/test_tasks.py     # anomaly detectors/metrics/tuning, survival columns, ID + leak guards
```

## Modules

| file | role |
|---|---|
| `intent.py` | prompt → `TaskSpec` (Gemini REST + `responseSchema`; rules fallback; column/task/metric validation; ID + survival column detection) |
| `data_prep.py` | leakage-free preprocessing (all fitting inside CV), outcome-leak checks, anomaly labels |
| `meta_features.py` | ~20–30 meta-features per task incl. landmarkers (label-free for anomaly detection) |
| `anomaly.py` | anomaly detectors (sklearn-style, higher score = more anomalous), metrics, meta-features |
| `benchmark.py` | re-runs every model on every dataset of a family → `artifacts/benchmarks/` |
| `dataset_specs.py` | per-dataset task overrides and skip list for the benchmark corpus |
| `knowledge_base.py` | joins meta-features with every (dataset, model, score, best_params) row |
| `ranker.py` | ranking meta-learner + LODO evaluation |
| `models.py` | model registry + conditional search spaces |
| `hpo.py` | CASH-style tuning of the top-k, multi-metric CV, out-of-fold anomaly scores |
| `forecasting.py` | lag-XGBoost / ARIMA / SARIMA / Prophet with rolling-origin CV |
