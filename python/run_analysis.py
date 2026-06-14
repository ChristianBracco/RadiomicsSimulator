from pathlib import Path
import json
import os

from core.dataset_loader import load_dataset
from core.patient_aggregation import aggregate_by_patient
from core.feature_pruning import correlation_pruning

from analysis.univariate_selection import select_features, normalize_selection_method, selection_method_label
from analysis.nested_logistic_cv import run_nested_cv
from analysis.permutation_test import permutation_test
from analysis.bootstrap_stability import bootstrap_stability
from analysis.loocv import run_loocv
from analysis.sample_size_simulation import sample_size_simulation
from analysis.sample_size_guardrail import build_sample_size_design_summary
from analysis.model_benchmark import compare_candidate_models

from train_final_model import train_final_model_from_data
from train_candidate_models import train_and_save_candidate_models



# ============================================================
# ENVIRONMENT OVERRIDES
# ============================================================

def _env_float(name, default):
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = float(str(raw).replace(",", "."))
    except ValueError:
        print(f"[WARN] Invalid {name}={raw!r}. Using default {default}.")
        return default
    return value


def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(float(str(raw).replace(",", ".")))
    except ValueError:
        print(f"[WARN] Invalid {name}={raw!r}. Using default {default}.")
        return default


def _env_optional_float(name, default=None):
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(str(raw).replace(",", "."))
    except ValueError:
        print(f"[WARN] Invalid {name}={raw!r}. Using default {default}.")
        return default


def _env_float_list(name, default):
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    out = []
    for part in str(raw).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part.replace(",", ".")))
        except ValueError:
            print(f"[WARN] Ignoring invalid value in {name}: {part!r}")
    return out or default


def _env_str(name, default):
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip()


def _env_str_list(name, default):
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    out = []
    for part in str(raw).replace(";", ",").split(","):
        part = part.strip()
        if part:
            out.append(part)
    return out or default

# ============================================================
# CONFIGURATION
# ============================================================

PRUNING_THRESHOLD = _env_float("THERADIOMICS_PRUNING_THRESHOLD", 0.85)
N_PERMUTATIONS = _env_int("THERADIOMICS_N_PERMUTATIONS", 500)
N_BOOTSTRAP = _env_int("THERADIOMICS_N_BOOTSTRAP", 500)
N_SAMPLE_SIZE_SIMULATIONS = _env_int("THERADIOMICS_N_SAMPLE_SIZE_SIMULATIONS", 100)
TOP_N_FINAL_MODEL_FEATURES = _env_int("THERADIOMICS_TOP_N_FINAL_MODEL_FEATURES", 1)
PRIMARY_MODEL_FEATURES = _env_int("THERADIOMICS_PRIMARY_MODEL_FEATURES", TOP_N_FINAL_MODEL_FEATURES)
FEATURE_SELECTION_METHOD = normalize_selection_method(_env_str("THERADIOMICS_FEATURE_SELECTION_METHOD", "lasso"))
SELECTION_METHODS_TO_COMPARE = [
    normalize_selection_method(x)
    for x in _env_str_list(
        "THERADIOMICS_SELECTION_METHODS_TO_COMPARE",
        ["pearson", "spearman", "auc", "mannwhitney", "mutual_info", "lasso"]
    )
]

# Analytical sample-size guardrail settings inspired by the
# shrinkage/event-rate framework for binary radiomics prediction models.
EXPECTED_EVENT_PREVALENCE = _env_optional_float("THERADIOMICS_EXPECTED_EVENT_PREVALENCE", None)  # None = use observed patient-level prevalence
RISK_MARGIN = _env_float("THERADIOMICS_RISK_MARGIN", 0.10)
TARGET_SHRINKAGE = _env_float("THERADIOMICS_TARGET_SHRINKAGE", 0.90)
DELTA_NAGELKERKE = _env_float("THERADIOMICS_DELTA_NAGELKERKE", 0.05)
EXPECTED_R2_CS_ADJ = _env_float("THERADIOMICS_EXPECTED_R2_CS_ADJ", 0.08)
SAMPLE_SIZE_RISK_MARGINS = _env_float_list("THERADIOMICS_SAMPLE_SIZE_RISK_MARGINS", [0.05, 0.075, 0.10])
SAMPLE_SIZE_R2_SCENARIOS = _env_float_list("THERADIOMICS_SAMPLE_SIZE_R2_SCENARIOS", [0.05, 0.08, 0.113])
SAMPLE_SIZE_PREDICTOR_COUNTS = [int(x) for x in _env_float_list("THERADIOMICS_SAMPLE_SIZE_PREDICTOR_COUNTS", [1, 2, 3, 5, 10])]

DATASET_FILENAME = "Features_all.xlsx"


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = (
    BASE_DIR.parent
    / "data"
    / "uploads"
)

INPUT_FILE = (
    UPLOAD_DIR
    / DATASET_FILENAME
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
)

MODELS_DIR = (
    BASE_DIR.parent
    / "models"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)

MODELS_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# UTILITIES
# ============================================================

def find_minimum_n_for_power(
    rows,
    target_power=0.80
):
    for row in rows:
        power = row.get("power")

        if power is not None and power >= target_power:
            return row["N"]

    return None


# ============================================================
# START
# ============================================================

print("\n============================================================")
print("THERADIOMICS STUDY DESIGNER")
print("============================================================")
print(f"Pruning threshold: {PRUNING_THRESHOLD}")
print(f"Feature selection method: {FEATURE_SELECTION_METHOD} ({selection_method_label(FEATURE_SELECTION_METHOD)})")
print(f"Final model features: {TOP_N_FINAL_MODEL_FEATURES}")

print("\nLooking for dataset:")
print(INPUT_FILE)

if not INPUT_FILE.exists():

    print("\nERROR: dataset not found.")
    print("\nExpected file:")
    print(INPUT_FILE)

    print("\nAvailable Excel files in upload folder:")

    if UPLOAD_DIR.exists():
        candidates = list(
            UPLOAD_DIR.glob("*.xlsx")
        )

        if candidates:
            for candidate in candidates:
                print(" -", candidate.name)
        else:
            print("No .xlsx files found.")
    else:
        print("Upload folder does not exist:")
        print(UPLOAD_DIR)

    raise FileNotFoundError(
        INPUT_FILE
    )


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\n------------------------------------------------------------")
print("1. LOADING DATASET")
print("------------------------------------------------------------")

df = load_dataset(
    INPUT_FILE
)

print(
    f"Lesions: {len(df)}"
)


# ============================================================
# 2. PATIENT-LEVEL AGGREGATION
# ============================================================

print("\n------------------------------------------------------------")
print("2. PATIENT-LEVEL AGGREGATION")
print("------------------------------------------------------------")

patient_df = aggregate_by_patient(
    df
)

print(
    f"Patients: {len(patient_df)}"
)


# ============================================================
# 3. PREPARE X, y, groups
# ============================================================

print("\n------------------------------------------------------------")
print("3. PREPARING X, y AND GROUPS")
print("------------------------------------------------------------")

y = patient_df[
    "BinaryOutcome"
]

groups = patient_df[
    "Patient"
]

X = patient_df.drop(
    columns=[
        "Patient",
        "BinaryOutcome"
    ]
)

print(
    f"Initial patient-level features: {X.shape[1]}"
)

print(
    f"Responders: {(y == 1).sum()}"
)

print(
    f"Non Responders: {(y == 0).sum()}"
)

observed_prevalence = float(
    (y == 1).mean()
)

prevalence_for_design = (
    float(EXPECTED_EVENT_PREVALENCE)
    if EXPECTED_EVENT_PREVALENCE is not None
    else observed_prevalence
)

print(
    f"Observed patient-level event prevalence: {observed_prevalence:.3f}"
)

print(
    f"Design prevalence used for sample-size guardrails: {prevalence_for_design:.3f}"
)


# ============================================================
# 4. CORRELATION PRUNING
# ============================================================

print("\n------------------------------------------------------------")
print("4. CORRELATION PRUNING")
print("------------------------------------------------------------")

print(
    f"Correlation threshold: {PRUNING_THRESHOLD}"
)

X_pruned, removed = correlation_pruning(
    X,
    threshold=PRUNING_THRESHOLD
)

print(
    f"Features before pruning: {X.shape[1]}"
)

print(
    f"Features after pruning: {X_pruned.shape[1]}"
)

print(
    f"Removed features: {len(removed)}"
)


# ============================================================
# 5. DESCRIPTIVE FEATURE SELECTION
# ============================================================

print("\n------------------------------------------------------------")
print("5. DESCRIPTIVE FEATURE SELECTION")
print("------------------------------------------------------------")
print(f"Primary selector: {FEATURE_SELECTION_METHOD} ({selection_method_label(FEATURE_SELECTION_METHOD)})")

selected = select_features(
    X_pruned,
    y,
    method=FEATURE_SELECTION_METHOD
)

selected_feature_selection_method_used = FEATURE_SELECTION_METHOD

if len(selected) == 0:
    print("[WARN] Primary selector returned zero features on the full dataset. Falling back to univariate AUC for final-model ranking.")
    selected = select_features(
        X_pruned,
        y,
        method="auc"
    )
    selected_feature_selection_method_used = "auc_fallback"

top_features = [
    item["feature"]
    for item in selected[:10]
]

print("\nTop selected features:")

for item in selected[:10]:
    metric = item.get("coef", item.get("correlation", item.get("oriented_auc", item.get("score"))))
    print(
        f" - {item['feature']} "
        f"(method={item.get('method', FEATURE_SELECTION_METHOD)}, score={item.get('score', None)}, metric={metric})"
    )


# ============================================================
# 6. NESTED CROSS VALIDATION
# ============================================================

print("\n------------------------------------------------------------")
print("6. NESTED CROSS VALIDATION")
print("------------------------------------------------------------")

cv = run_nested_cv(
    X_pruned,
    y,
    groups,
    n_splits=5,
    top_n_features=PRIMARY_MODEL_FEATURES,
    selection_method=FEATURE_SELECTION_METHOD,
    return_fold_details=True,
    verbose=True
)

print(
    f"Nested CV AUC: {cv['mean_auc']:.3f} ± {cv['std_auc']:.3f}"
)


# ============================================================
# 7. PERMUTATION TEST
# ============================================================

print("\n------------------------------------------------------------")
print("7. PERMUTATION TEST")
print("------------------------------------------------------------")

perm = permutation_test(
    X_pruned,
    y,
    groups,
    cv["mean_auc"],
    n_permutations=N_PERMUTATIONS,
    top_n_features=PRIMARY_MODEL_FEATURES,
    selection_method=FEATURE_SELECTION_METHOD
)

print(
    f"Observed AUC: {perm['observed_auc']:.3f}"
)

print(
    f"Mean random AUC: {perm['mean_random_auc']:.3f}"
)

print(
    f"Max random AUC: {perm['max_random_auc']:.3f}"
)

print(
    f"P-value: {perm['p_value']}"
)


# ============================================================
# 8. BOOTSTRAP STABILITY
# ============================================================

print("\n------------------------------------------------------------")
print("8. BOOTSTRAP STABILITY")
print("------------------------------------------------------------")

bootstrap = bootstrap_stability(
    X_pruned,
    y,
    n_bootstrap=N_BOOTSTRAP,
    top_n_features=10,
    selection_method=FEATURE_SELECTION_METHOD
)

print("\nTop bootstrap-stable features:")

for item in bootstrap[:10]:
    print(
        f" - {item['feature']} "
        f"({item['frequency']:.1%})"
    )


# ============================================================
# 9. LEAVE-ONE-OUT CROSS VALIDATION
# ============================================================

print("\n------------------------------------------------------------")
print("9. LEAVE-ONE-OUT CROSS VALIDATION")
print("------------------------------------------------------------")

loocv = run_loocv(
    X_pruned,
    y,
    top_n_features=PRIMARY_MODEL_FEATURES,
    selection_method=FEATURE_SELECTION_METHOD
)

print(
    f"LOOCV AUC: {loocv['auc']:.3f}"
)

print(
    f"LOOCV Sensitivity: {loocv['sensitivity']:.3f}"
)

print(
    f"LOOCV Specificity: {loocv['specificity']:.3f}"
)


# ============================================================
# 10. CANDIDATE MODEL BENCHMARK
# ============================================================

print("\n------------------------------------------------------------")
print("10. CANDIDATE MODEL BENCHMARK")
print("------------------------------------------------------------")

candidate_models = compare_candidate_models(
    X=X_pruned,
    y=y,
    groups=groups,
    selected_features_detail=selected,
    bootstrap_features_detail=bootstrap,
    n_splits=5,
    return_details=True,
    selection_method=FEATURE_SELECTION_METHOD,
    selection_methods_to_compare=SELECTION_METHODS_TO_COMPARE
)

print("\nCandidate model comparison:")

for model in candidate_models["summary"]:
    if model.get("status") != "ok":
        print(
            f" - {model['label']}: skipped"
        )
        continue

    print(
        f" - {model['label']} | "
        f"features={model['n_features']} | "
        f"GroupCV AUC={model['group_cv_mean_auc']:.3f} | "
        f"LOOCV AUC={model['loocv_auc']:.3f}"
    )


# ============================================================
# 11. TRAIN AND SAVE FINAL MODEL
# ============================================================

print("\n------------------------------------------------------------")
print("10. TRAINING FINAL DEPLOYABLE MODEL")
print("------------------------------------------------------------")

validation_summary = {
    "nested_cv": cv,
    "loocv": loocv,
    "permutation": perm,
    "note": (
        "These are validation metrics. The final model is trained on all "
        "available patients only after validation is complete."
    )
}

dataset_summary = {
    "lesions": int(len(df)),
    "patients": int(len(patient_df)),
    "responders": int((y == 1).sum()),
    "non_responders": int((y == 0).sum()),
    "observed_prevalence": float(observed_prevalence),
    "design_prevalence": float(prevalence_for_design),
    "primary_model_features": int(PRIMARY_MODEL_FEATURES),
    "feature_selection_method": FEATURE_SELECTION_METHOD,
    "selected_feature_selection_method_used": selected_feature_selection_method_used,
    "features_before_pruning": int(X.shape[1]),
    "features_after_pruning": int(X_pruned.shape[1])
}

model_info = train_final_model_from_data(
    X_pruned=X_pruned,
    y=y,
    selected_features_detail=selected,
    removed_features=removed,
    input_file=INPUT_FILE,
    models_dir=MODELS_DIR,
    pruning_threshold=PRUNING_THRESHOLD,
    top_n_features=TOP_N_FINAL_MODEL_FEATURES,
    dataset_summary=dataset_summary,
    validation_summary=validation_summary,
    feature_selection_method=selected_feature_selection_method_used
)

print("Final model saved:")
print(model_info["model_path"])
print("Model metadata saved:")
print(model_info["metadata_path"])
print(
    f"Apparent training AUC: {model_info['apparent_training_auc']:.3f}"
)

print("\nTraining and saving candidate models...")

saved_candidate_models = train_and_save_candidate_models(
    X_pruned=X_pruned,
    y=y,
    candidate_models=candidate_models["models"],
    models_dir=MODELS_DIR,
    threshold=PRUNING_THRESHOLD,
    input_file=INPUT_FILE,
    removed_features=removed,
    dataset_summary=dataset_summary,
    validation_summary=validation_summary
)

print("Candidate models metadata saved:")
print(saved_candidate_models["collection_metadata_path"])

for saved_model in saved_candidate_models["models"]:
    if saved_model.get("status") == "saved":
        print(
            f" - {saved_model['label']}: {saved_model['model_path']}"
        )


# ============================================================
# 12. SAMPLE SIZE SIMULATION
# ============================================================

print("\n------------------------------------------------------------")
print("12. SAMPLE SIZE SIMULATION")
print("------------------------------------------------------------")

sample_size_common_kwargs = {
    "n_simulations": N_SAMPLE_SIZE_SIMULATIONS,
    "prevalence": prevalence_for_design,
    "risk_margin": RISK_MARGIN,
    "r2_cs_adj": EXPECTED_R2_CS_ADJ,
    "final_predictors_k": PRIMARY_MODEL_FEATURES,
    "target_shrinkage": TARGET_SHRINKAGE,
    "delta_nagelkerke": DELTA_NAGELKERKE
}

sample_size = {
    "auc_070": sample_size_simulation(
        0.70,
        scenario_label="theoretical_auc_070",
        **sample_size_common_kwargs
    ),
    "auc_080": sample_size_simulation(
        0.80,
        scenario_label="theoretical_auc_080",
        **sample_size_common_kwargs
    ),
    "auc_090": sample_size_simulation(
        0.90,
        scenario_label="theoretical_auc_090",
        **sample_size_common_kwargs
    ),
    "observed_nested_cv_auc": sample_size_simulation(
        float(cv["mean_auc"]),
        scenario_label="observed_nested_cv_auc",
        **sample_size_common_kwargs
    ),
    "observed_loocv_auc": sample_size_simulation(
        float(loocv["auc"]),
        scenario_label="observed_loocv_auc",
        **sample_size_common_kwargs
    )
}

sample_size_design = build_sample_size_design_summary(
    current_n=int(len(patient_df)),
    prevalence=prevalence_for_design,
    risk_margins=SAMPLE_SIZE_RISK_MARGINS,
    r2_cs_adj_scenarios=SAMPLE_SIZE_R2_SCENARIOS,
    predictor_counts=SAMPLE_SIZE_PREDICTOR_COUNTS,
    target_shrinkage=TARGET_SHRINKAGE,
    delta_nagelkerke=DELTA_NAGELKERKE
)


# ============================================================
# 13. FINAL RESULT OBJECT
# ============================================================

result = {
    "patients": int(len(patient_df)),
    "lesions": int(len(df)),
    "responders": int((y == 1).sum()),
    "non_responders": int((y == 0).sum()),
    "observed_prevalence": float(observed_prevalence),
    "design_prevalence": float(prevalence_for_design),
    "primary_model_features": int(PRIMARY_MODEL_FEATURES),
    "feature_selection_method": FEATURE_SELECTION_METHOD,
    "selected_feature_selection_method_used": selected_feature_selection_method_used,
    "pruning_threshold": PRUNING_THRESHOLD,
    "run_configuration": {
        "pruning_threshold": float(PRUNING_THRESHOLD),
        "pruning_threshold_source": "env" if os.environ.get("THERADIOMICS_PRUNING_THRESHOLD") else "default",
        "top_n_final_model_features": int(TOP_N_FINAL_MODEL_FEATURES),
        "primary_model_features": int(PRIMARY_MODEL_FEATURES),
        "feature_selection_method": FEATURE_SELECTION_METHOD,
        "selected_feature_selection_method_used": selected_feature_selection_method_used,
        "selection_methods_compared": SELECTION_METHODS_TO_COMPARE,
        "n_permutations": int(N_PERMUTATIONS),
        "n_bootstrap": int(N_BOOTSTRAP),
        "n_sample_size_simulations": int(N_SAMPLE_SIZE_SIMULATIONS),
        "risk_margin": float(RISK_MARGIN),
        "expected_r2_cs_adj": float(EXPECTED_R2_CS_ADJ),
        "target_shrinkage": float(TARGET_SHRINKAGE),
        "delta_nagelkerke": float(DELTA_NAGELKERKE),
    },
    "features_before_pruning": int(X.shape[1]),
    "features_after_pruning": int(X_pruned.shape[1]),
    "removed_features_count": int(len(removed)),
    "removed_features": removed,
    "selected_features": selected[:20],
    "top_features": top_features,
    "cv": cv,
    "nested_cv_debug_file": str(OUTPUT_DIR / "nested_cv_fold_details.json"),
    "permutation": perm,
    "bootstrap": bootstrap[:20],
    "loocv": loocv,
    "sample_size": sample_size,
    "sample_size_design": sample_size_design,
    "candidate_models": candidate_models,
    "candidate_models_file": str(OUTPUT_DIR / "candidate_model_comparison.json"),
    "final_model": model_info,
    "saved_candidate_models": saved_candidate_models,
    "model_explanation": {
        "classifier": "StandardScaler + LogisticRegression",
        "feature_selection": selected_feature_selection_method_used,
        "saved_model": model_info["model_path"],
        "saved_metadata": model_info["metadata_path"],
        "important_note": (
            "The saved final model is trained on all available patients. "
            "Its apparent training AUC is optimistic. Use Nested CV, LOOCV "
            "and permutation test as validation metrics."
        )
    }
}


# ============================================================
# 14. SAVE JSON
# ============================================================

json_path = (
    OUTPUT_DIR
    / "analysis_results.json"
)

with open(
    json_path,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        result,
        f,
        indent=2,
        ensure_ascii=False
    )

print("\nSaved JSON:")
print(json_path)

# Dedicated debug JSON for HTML visualization and threshold comparison
nested_cv_debug_path = OUTPUT_DIR / "nested_cv_fold_details.json"

with open(
    nested_cv_debug_path,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        {
            "pruning_threshold": PRUNING_THRESHOLD,
            "mean_auc": cv["mean_auc"],
            "std_auc": cv["std_auc"],
            "folds": cv["folds"],
            "stable_features": cv["stable_features"],
            "all_predictions": cv.get("all_predictions", []),
            "fold_details": cv.get("fold_details", []),
            "debug_note": cv.get("debug_note", "")
        },
        f,
        indent=2,
        ensure_ascii=False
    )

print("Saved nested CV debug JSON:")
print(nested_cv_debug_path)

candidate_models_path = OUTPUT_DIR / "candidate_model_comparison.json"

with open(
    candidate_models_path,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        candidate_models,
        f,
        indent=2,
        ensure_ascii=False
    )

print("Saved candidate model comparison JSON:")
print(candidate_models_path)


# ============================================================
# 15. SAVE TEXT REPORT
# ============================================================

summary_path = (
    OUTPUT_DIR
    / "summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write("============================================================\n")
    f.write("THERADIOMICS STUDY DESIGNER\n")
    f.write("============================================================\n\n")

    f.write("DATASET\n")
    f.write("-------\n")
    f.write(f"Input file: {INPUT_FILE}\n")
    f.write(f"Lesions: {len(df)}\n")
    f.write(f"Patients: {len(patient_df)}\n")
    f.write(f"Responders: {(y == 1).sum()}\n")
    f.write(f"Non Responders: {(y == 0).sum()}\n\n")

    f.write("FEATURE PRUNING\n")
    f.write("---------------\n")
    f.write(f"Correlation threshold: {PRUNING_THRESHOLD}\n")
    f.write(f"Features before pruning: {X.shape[1]}\n")
    f.write(f"Features after pruning: {X_pruned.shape[1]}\n")
    f.write(f"Removed features: {len(removed)}\n\n")

    f.write("MODEL PERFORMANCE\n")
    f.write("-----------------\n")
    f.write(f"Nested CV AUC: {cv['mean_auc']:.3f} ± {cv['std_auc']:.3f}\n")
    f.write(f"Nested CV debug file: {OUTPUT_DIR / 'nested_cv_fold_details.json'}\n")
    f.write(f"LOOCV AUC: {loocv['auc']:.3f}\n")
    f.write(f"LOOCV Sensitivity: {loocv['sensitivity']:.3f}\n")
    f.write(f"LOOCV Specificity: {loocv['specificity']:.3f}\n\n")

    f.write("CANDIDATE MODEL BENCHMARK\n")
    f.write("-------------------------\n")

    for model in candidate_models["summary"]:
        if model.get("status") != "ok":
            f.write(f"{model['label']}: skipped\n")
            continue

        f.write(
            f"{model['label']} | "
            f"features={model['n_features']} | "
            f"GroupCV AUC={model['group_cv_mean_auc']:.3f} ± {model['group_cv_std_auc']:.3f} | "
            f"LOOCV AUC={model['loocv_auc']:.3f} | "
            f"Sens={model['loocv_sensitivity']:.3f} | "
            f"Spec={model['loocv_specificity']:.3f}\n"
        )

        f.write("    Features: " + ", ".join(model["features"]) + "\n")

    f.write("\nCandidate model comparison JSON: ")
    f.write(str(OUTPUT_DIR / "candidate_model_comparison.json") + "\n")
    f.write("Candidate model files metadata: ")
    f.write(saved_candidate_models["collection_metadata_path"] + "\n\n")

    f.write("PERMUTATION TEST\n")
    f.write("----------------\n")
    f.write(f"Observed AUC: {perm['observed_auc']:.3f}\n")
    f.write(f"Mean Random AUC: {perm['mean_random_auc']:.3f}\n")
    f.write(f"Max Random AUC: {perm['max_random_auc']:.3f}\n")

    if perm["p_value"] == 0:
        f.write(
            f"P-value: < {1 / N_PERMUTATIONS:.4f} "
            f"(0/{N_PERMUTATIONS} permutations exceeded observed AUC)\n\n"
        )
    else:
        f.write(f"P-value: {perm['p_value']}\n\n")

    f.write("FINAL SAVED MODEL\n")
    f.write("-----------------\n")
    f.write("Classifier: StandardScaler + LogisticRegression\n")
    f.write(f"Feature selection: {selected_feature_selection_method_used}\n")
    f.write(f"Model file: {model_info['model_path']}\n")
    f.write(f"Metadata file: {model_info['metadata_path']}\n")
    f.write(f"Apparent training AUC: {model_info['apparent_training_auc']:.3f}\n")
    f.write("Warning: apparent training AUC is optimistic and is not validation performance.\n\n")

    f.write("TOP FINAL MODEL FEATURES\n")
    f.write("------------------------\n")

    for feature in model_info["selected_features"]:
        f.write(f"{feature}\n")

    f.write("\nTOP SELECTED FEATURES\n")
    f.write("---------------------\n")

    for item in selected[:10]:
        metric = item.get("coef", item.get("correlation", item.get("oriented_auc", item.get("score"))))
        f.write(
            f"{item['feature']} "
            f"(method={item.get('method', selected_feature_selection_method_used)}, score={item.get('score')}, metric={metric})\n"
        )

    f.write("\nBOOTSTRAP STABILITY\n")
    f.write("-------------------\n")

    for item in bootstrap[:10]:
        f.write(
            f"{item['feature']} "
            f"({item['frequency']:.1%})\n"
        )

    f.write("\nSTABLE FEATURES IN LOOCV\n")
    f.write("------------------------\n")

    for feature, count in loocv["stable_features"][:10]:
        f.write(f"{feature}: {count} folds\n")

    f.write("\nSAMPLE SIZE ESTIMATION\n")
    f.write("----------------------\n")

    scenario_order = [
        ("auc_070", "Theoretical AUC 0.70"),
        ("auc_080", "Theoretical AUC 0.80"),
        ("auc_090", "Theoretical AUC 0.90"),
        ("observed_nested_cv_auc", f"Observed Nested CV AUC {cv['mean_auc']:.3f}"),
        ("observed_loocv_auc", f"Observed LOOCV AUC {loocv['auc']:.3f}")
    ]

    for key, label in scenario_order:
        f.write(f"\nScenario: {label}\n")

        target_n = find_minimum_n_for_power(
            result["sample_size"][key]
        )

        for row in result["sample_size"][key]:
            mean_auc = row.get("mean_auc")
            power = row.get("power")

            auc_text = "NA" if mean_auc is None else f"{mean_auc:.3f}"
            power_text = "NA" if power is None else f"{power:.1%}"

            f.write(
                f"N={row['N']:>3} | "
                f"Events={row.get('expected_events', 'NA'):>3} | "
                f"Non-events={row.get('expected_nonevents', 'NA'):>3} | "
                f"AUC={auc_text} | "
                f"POWER={power_text} | "
                f"k_max={row.get('k_max_raw', 0):.2f} | "
                f"Design={row.get('design_label', 'NA')}\n"
            )

        if target_n is not None:
            f.write(
                f"\nMinimum sample size for 80% power: {target_n} patients\n"
            )
        else:
            f.write(
                "\nMinimum sample size for 80% power: not reached in simulated range\n"
            )

    f.write("\nANALYTICAL SAMPLE-SIZE GUARDRAILS\n")
    f.write("---------------------------------\n")
    f.write(f"Observed prevalence: {observed_prevalence:.3f}\n")
    f.write(f"Design prevalence: {prevalence_for_design:.3f}\n")
    f.write(f"Primary model features: {PRIMARY_MODEL_FEATURES}\n")
    f.write(f"Target shrinkage: {TARGET_SHRINKAGE:.2f}\n")
    f.write(f"Expected R2 Cox-Snell adjusted scenario: {EXPECTED_R2_CS_ADJ:.3f}\n\n")

    f.write("Event-rate precision guardrails:\n")
    for item in sample_size_design["event_rate_guardrails"]:
        f.write(
            f" - Margin ±{item['risk_margin']:.3f}: "
            f"N required = {item['n_required']}\n"
        )

    f.write("\nPredictor/shrinkage guardrails by R2 scenario:\n")
    for block in sample_size_design["predictor_guardrails"]:
        f.write(
            f"\nR2_CS_adj = {block['r2_cs_adj']:.3f} | "
            f"k_max at current N = {block['k_max_current_n']['k_max_raw']:.2f} "
            f"(floor {block['k_max_current_n']['k_max_floor']})\n"
        )
        for item in block["n_required_by_predictor_count"]:
            f.write(
                f"   - k={item['final_predictors_k']}: "
                f"N shrinkage required = {item['n_required_shrinkage']}\n"
            )

    f.write(
        "\nInterpretation: 150-200 independent patients are plausible only "
        "for a very compact signature, ideally 1-2 final predictors. "
        "Larger feature sets remain exploratory or require larger cohorts.\n"
    )

print("Saved summary:")
print(summary_path)


# ============================================================
# 16. FINAL CONSOLE SUMMARY
# ============================================================

print("\n============================================================")
print("THERADIOMICS STUDY COMPLETED")
print("============================================================")
print(f"Input file: {INPUT_FILE}")
print(f"Lesions: {len(df)}")
print(f"Patients: {len(patient_df)}")
print(f"Responders: {(y == 1).sum()}")
print(f"Non Responders: {(y == 0).sum()}")
print(f"Observed prevalence: {observed_prevalence:.3f}")
print(f"Design prevalence: {prevalence_for_design:.3f}")
print(f"Primary model features: {PRIMARY_MODEL_FEATURES}")
print(f"Feature selection method: {FEATURE_SELECTION_METHOD}")
print(f"Pruning threshold: {PRUNING_THRESHOLD}")
print(f"Features before pruning: {X.shape[1]}")
print(f"Features after pruning: {X_pruned.shape[1]}")
print(f"Nested CV AUC: {cv['mean_auc']:.3f} ± {cv['std_auc']:.3f}")
print(f"LOOCV AUC: {loocv['auc']:.3f}")
print(f"Permutation p-value: {perm['p_value']}")
print(f"Final model: {model_info['model_path']}")
print(f"Metadata: {model_info['metadata_path']}")
print(f"Candidate models: {saved_candidate_models['collection_metadata_path']}")
print(f"Candidate model comparison: {OUTPUT_DIR / 'candidate_model_comparison.json'}")
print(f"Nested CV debug: {OUTPUT_DIR / 'nested_cv_fold_details.json'}")
print("============================================================")
