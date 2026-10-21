"""Hand-checked task definitions for the benchmark datasets under Datasets/.

The knowledge base must describe each dataset with the problem its benchmark score was measured
on, and keyword heuristics get several of these wrong (e.g. METABRIC's "status" column is an ER
receptor test, mgus has two time columns). Anything listed here overrides the heuristics in
knowledge_base.benchmark_spec; datasets in SKIP are left out of benchmarks and the KB.

Override keys: time, event, event_value (the value that means "event happened"), target,
exclude (columns that leak the outcome or are identifiers), query (row filter, pandas syntax).
"""
from __future__ import annotations

OVERRIDES: dict[str, dict[str, dict]] = {
    "survival": {
        "METABRIC_RNA_Mutation.csv": dict(
            time="overall_survival_months", event="overall_survival", event_value=0,   # 1 = living
            exclude=["death_from_cancer", "patient_id"]),
        "Clinical Data_Discovery_Cohort.csv": dict(
            time="Time", event="Event",
            exclude=["Dead or Alive", "Date of Death", "Date of Last Follow Up", "Specimen date", "PatientID"]),
        "Standard Survival Data Format.csv": dict(
            time="Remission Time (t)", event="Censoring (d)", exclude=["Patient (i)"]),   # IDs sorted by time
        "colon.csv": dict(time="time", event="status", query="etype == 2",   # one row per patient: death
                          exclude=["etype", "id", "study"]),
        "mgus.csv": dict(time="futime", event="death", exclude=["pctime", "pcdx", "id"]),  # pc* = later progression
        "myeloid.csv": dict(time="futime", event="death", exclude=["txtime", "crtime", "rltime", "id"]),
        "lung.csv": dict(time="time", event="status", event_value=2),
        "kidney.csv": dict(time="time", event="status", exclude=["id"]),
        "myeloma.csv": dict(time="futime", event="death", exclude=["id"]),
        # added from Rdatasets / scikit-survival (fetched by scripts/fetch_kb_datasets.py)
        "cirrhosis_pbc.csv": dict(time="N_Days", event="Status", event_value="D", exclude=["ID"]),  # CL = transplant
        "r_Melanoma.csv": dict(time="time", event="status", event_value=1),     # 2 alive, 3 died of other causes
        "r_burn.csv": dict(time="T3", event="D3", exclude=["Obs", "T1", "D1", "T2", "D2"]),  # T1/T2 = later treatments
        "r_flchain.csv": dict(time="futime", event="death", exclude=["chapter"]),  # chapter = cause of death
        "r_hodg.csv": dict(time="time", event="delta"),
        "r_kidtran.csv": dict(time="time", event="delta", exclude=["obs"]),
        "r_nwtco.csv": dict(time="edrel", event="rel", exclude=["seqno", "in.subcohort"]),
        "r_pneumon.csv": dict(time="chldage", event="hospital", exclude=["agepn"]),
        "r_std.csv": dict(time="time", event="rinfct", exclude=["obs"]),
    },
    "classification": {
        "NHANES_age_prediction.csv": dict(target="age_group", exclude=["SEQN", "RIDAGEYR"]),   # RIDAGEYR = age
        "breast_wisconsin_.csv": dict(target="diagnosis", exclude=["id"]),
        "heart_failure_clinical_records_dataset.csv": dict(target="DEATH_EVENT", exclude=["time"]),  # follow-up time
        "heart_disease_uci.csv": dict(target="num", exclude=["id"]),
    },
    "regression": {
        "Life Expectancy Data.csv": dict(target="Life expectancy", exclude=["Country"]),
        "bodyfat.csv": dict(target="BodyFat", exclude=["Density"]),   # body fat is computed from density (Siri)
    },
}

SKIP: dict[str, dict[str, str]] = {
    "survival": {
        "cancer.csv": "byte-identical duplicate of lung.csv",
        "haberman.csv": "no survival-time column (only 5-year status); benchmarked as classification instead",
        "survival_prediction.csv": "in-hospital death classification data; no survival time",
    },
    "classification": {
        "Indian Liver Patient Dataset (ILPD).csv": "duplicate of indian_liver_patient.csv, without a header row",
    },
    "regression": {
        "cirrhosis.csv": "Mayo PBC trial: benchmarked as survival (cirrhosis_pbc.csv); 'Stage' is ordinal",
        "diabetes.csv": "0/1 target: a classification dataset (Classification/diabetes.csv)",
        "healthcare-dataset-stroke-data.csv": "0/1 target: a classification dataset",
        "heart_failure_clinical_records_dataset.csv": "0/1 target: a classification dataset",
        "hypertension_data.csv": "0/1 target: a classification dataset",
        "healthcare_dataset.csv": "synthetic data with a categorical target and no signal",
        "parkinsons_updrs.csv": "~140 repeated visits per patient: random CV folds leak patient identity",
    },
    "timeseries": {
        "Blood Transfusion Service data.csv": "tabular classification data, not a time series",
        "Clinical Data_Discovery_Cohort.csv": "survival table, not a time series",
        "Opthalmology.csv": "eye-measurement table, not a time series",
        "Opthalmology2.csv": "duplicate of Opthalmology.csv",
        "Vital_signs.csv": "per-surgery summary table, not a time series",
        "gait1.csv": "target was the activity label; replaced by derived_gait1_ankle_vertical.csv",
        "gait2.csv": "target was the activity label; replaced by derived_gait2_ankle_vertical.csv",
        "gait3.csv": "target was the activity label; replaced by derived_gait3_ankle_vertical.csv",
        "heart_rate.csv": "4 separate recordings; replaced by derived_heart_rate_T1..T4.csv",
        "ecg_1d_timeseries_prediction.csv": "semicolon-separated; replaced by derived_ecg.csv",
        "human_vital_signs_dataset_2024.csv": "per-patient table with a categorical target",
        "scores.csv": "depression-study table, not a time series",
        "wearable_sports_health_dataset.csv": "categorical target, not a time series",
    },
}


def override(family: str, dataset: str) -> dict:
    return OVERRIDES.get(family, {}).get(dataset, {})


def skipped(family: str, dataset: str) -> str | None:
    return SKIP.get(family, {}).get(dataset)
