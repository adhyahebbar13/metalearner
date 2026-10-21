"""Leave-one-dataset-out evaluation of the meta-learner ranking vs. simple baselines.

    python scripts/evaluate_metalearner.py            # uses artifacts/knowledge_base.csv (built if missing)
    python scripts/evaluate_metalearner.py --rebuild  # recompute meta-features from Datasets/ first

regret = (score of best model - score of chosen model) / (best - worst) on that dataset; 0 is perfect.
baseline = the model with the lowest average regret on the other datasets ("always pick the usual winner").
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metalearner.knowledge_base import build, load_kb  # noqa: E402
from metalearner.ranker import evaluate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rebuild", action="store_true")
a = ap.parse_args()
kb = build() if a.rebuild else load_kb()

rows = []
for fam in ["classification", "regression", "survival", "timeseries", "anomaly"]:
    if fam not in set(kb.family):
        continue
    r = evaluate(kb, fam)
    rows.append({"task": fam, "datasets": r["n_datasets"],
                 "meta top-1 hit": r["top1_hit"], "meta top-2 hit": r["top2_hit"],
                 "baseline top-1 hit": r["baseline_top1_hit"],
                 "meta regret (top-1)": r["regret_top1"], "meta regret (best of top-2)": r["regret_best_of_top2"],
                 "baseline regret": r["baseline_regret"],
                 "baseline regret (best of top-2)": r["baseline_regret_best_of_top2"],
                 "random regret": r["random_regret"]})
df = pd.DataFrame(rows).set_index("task")
pd.set_option("display.width", 200)
print(df.round(3).to_string())
