from pathlib import Path
import json
import math
from datetime import datetime

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.dataset_loader import load_dataset
from core.patient_aggregation import aggregate_by_patient
from core.feature_pruning import correlation_pruning
from analysis.lasso_selection import lasso_selection


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_PRUNING_THRESHOLD = 0.85
DEFAULT_TOP_N_FEATURES = 1
DATASET_FILENAME = "Features_all.xlsx"


# ============================================================
# JSON HELPERS
# ============================================================

def _to_jsonable(value):
    """
    Convert numpy / sklearn-friendly objects into JSON-safe values.
    """

    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]

    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(value, Path):
        return str(value)

    return value


# ============================================================
# PATH HELPERS
# ============================================================

def get_default_paths():
    """
    Expected project layout when this file is placed in:

    server/server/python/train_final_model.py

    Expected dataset:
    server/server/data/uploads/Features_all.xlsx

    Output models:
    server/server/models/
    """

    base_dir = Path(__file__).resolve().parent

    upload_dir = (
        base_dir.parent
        / "data"
        / "uploads"
    )

    input_file = (
        upload_dir
        / DATASET_FILENAME
    )

    models_dir = (
        base_dir.parent
        / "models"
    )

    return base_dir, upload_dir, input_file, models_dir


# ============================================================
# CORE TRAINING FUNCTION
# ============================================================

def train_final_model_from_data(
    X_pruned,
    y,
    selected_features_detail,
    removed_features,
    input_file,
    models_dir,
    pruning_threshold=DEFAULT_PRUNING_THRESHOLD,
    top_n_features=DEFAULT_TOP_N_FEATURES,
    dataset_summary=None,
    validation_summary=None,
    feature_selection_method="lasso",
):
    """
    Train the final deployable binary classifier on all available patients.

    Important:
    - this model is NOT used to estimate performance;
    - performance must come from Nested CV / LOOCV / permutation test;
    - this final model is the object to save and reuse for future prediction.
    """

    models_dir = Path(models_dir)
    models_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    selected_features = [
        item["feature"]
        for item in selected_features_detail[:top_n_features]
    ]

    if len(selected_features) == 0:
        raise ValueError(
            "No selected features available. Cannot train final model."
        )

    missing = [
        f for f in selected_features
        if f not in X_pruned.columns
    ]

    if missing:
        raise ValueError(
            "Selected features missing from X_pruned: "
            + ", ".join(missing)
        )

    X_final = X_pruned[
        selected_features
    ]

    pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LogisticRegression(
                    solver="liblinear",
                    l1_ratio=0.0,
                    C=1.0,
                    max_iter=5000
                )
            )
        ]
    )

    pipeline.fit(
        X_final,
        y
    )

    train_probabilities = pipeline.predict_proba(
        X_final
    )[:, 1]

    apparent_training_auc = roc_auc_score(
        y,
        train_probabilities
    )

    classifier = pipeline.named_steps[
        "classifier"
    ]

    model_coefficients = []

    for feature, coef in zip(
        selected_features,
        classifier.coef_[0]
    ):
        model_coefficients.append({
            "feature": feature,
            "coef": float(coef)
        })

    threshold_label = str(pruning_threshold).replace(
        ".",
        ""
    )

    model_filename = (
        f"theradiomics_threshold_{threshold_label}.joblib"
    )

    metadata_filename = (
        f"theradiomics_threshold_{threshold_label}_metadata.json"
    )

    model_path = (
        models_dir
        / model_filename
    )

    metadata_path = (
        models_dir
        / metadata_filename
    )

    model_package = {
        "pipeline": pipeline,
        "selected_features": selected_features,
        "feature_columns_after_pruning": list(X_pruned.columns),
        "removed_features": list(removed_features),
        "pruning_threshold": float(pruning_threshold),
        "top_n_features": int(top_n_features),
        "input_file": str(input_file),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_type": "StandardScaler + LogisticRegression",
        "feature_selection": f"{feature_selection_method} before final fit"
    }

    joblib.dump(
        model_package,
        model_path
    )

    metadata = {
        "created_at": model_package["created_at"],
        "model_file": str(model_path),
        "metadata_file": str(metadata_path),
        "input_file": str(input_file),
        "model_type": "StandardScaler + LogisticRegression",
        "classifier": {
            "name": "LogisticRegression",
            "solver": "liblinear",
            "C": 1.0,
            "max_iter": 5000
        },
        "feature_selection": {
            "method": feature_selection_method,
            "top_n_features": int(top_n_features),
            "selected_features": selected_features,
            "selected_features_detail": selected_features_detail[:top_n_features]
        },
        "pruning": {
            "threshold": float(pruning_threshold),
            "features_after_pruning": int(X_pruned.shape[1]),
            "removed_features_count": int(len(removed_features)),
            "removed_features": list(removed_features)
        },
        "final_model": {
            "trained_on_all_available_patients": True,
            "apparent_training_auc": float(apparent_training_auc),
            "intercept": float(classifier.intercept_[0]),
            "coefficients": model_coefficients,
            "note": (
                "apparent_training_auc is optimistic and must not be used "
                "as validation performance. Use Nested CV / LOOCV / permutation."
            )
        },
        "dataset_summary": dataset_summary or {},
        "validation_summary": validation_summary or {}
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            _to_jsonable(metadata),
            f,
            indent=2,
            ensure_ascii=False
        )

    return {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "selected_features": selected_features,
        "apparent_training_auc": float(apparent_training_auc),
        "coefficients": model_coefficients,
        "intercept": float(classifier.intercept_[0])
    }


# ============================================================
# STANDALONE ENTRY POINT
# ============================================================

def main():

    base_dir, upload_dir, input_file, models_dir = get_default_paths()

    print("\n============================================================")
    print("THERADIOMICS FINAL MODEL TRAINING")
    print("============================================================")
    print("\nLooking for dataset:")
    print(input_file)

    if not input_file.exists():
        raise FileNotFoundError(
            input_file
        )

    print("\nLoading dataset...")
    df = load_dataset(
        input_file
    )

    print("Aggregating by patient...")
    patient_df = aggregate_by_patient(
        df
    )

    y = patient_df[
        "BinaryOutcome"
    ]

    X = patient_df.drop(
        columns=[
            "Patient",
            "BinaryOutcome"
        ]
    )

    print("Applying correlation pruning...")
    X_pruned, removed = correlation_pruning(
        X,
        threshold=DEFAULT_PRUNING_THRESHOLD
    )

    print("Running LASSO feature selection...")
    selected = lasso_selection(
        X_pruned,
        y
    )

    dataset_summary = {
        "lesions": int(len(df)),
        "patients": int(len(patient_df)),
        "responders": int((y == 1).sum()),
        "non_responders": int((y == 0).sum()),
        "features_before_pruning": int(X.shape[1]),
        "features_after_pruning": int(X_pruned.shape[1])
    }

    info = train_final_model_from_data(
        X_pruned=X_pruned,
        y=y,
        selected_features_detail=selected,
        removed_features=removed,
        input_file=input_file,
        models_dir=models_dir,
        pruning_threshold=DEFAULT_PRUNING_THRESHOLD,
        top_n_features=DEFAULT_TOP_N_FEATURES,
        dataset_summary=dataset_summary,
        validation_summary={}
    )

    print("\nFinal model saved:")
    print(info["model_path"])
    print("\nMetadata saved:")
    print(info["metadata_path"])
    print("\nSelected features:")

    for feature in info["selected_features"]:
        print(" -", feature)

    print("\nApparent training AUC:", f"{info['apparent_training_auc']:.3f}")
    print("============================================================")


if __name__ == "__main__":
    main()
