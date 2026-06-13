import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from analysis.lasso_selection import lasso_selection


# ============================================================
# JSON HELPERS
# ============================================================

def _json_scalar(value):
    """
    Convert numpy / pandas scalar values into JSON-safe Python values.
    """

    if pd.isna(value):
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    return value


def _value_counts_as_dict(series):
    """
    Return value counts as a JSON-safe dictionary.
    """

    counts = series.value_counts().to_dict()

    return {
        str(_json_scalar(k)): int(v)
        for k, v in counts.items()
    }


# ============================================================
# NESTED / FOLD-WISE VALIDATION WITH DEBUG DETAILS
# ============================================================

def run_nested_cv(
    X,
    y,
    groups,
    n_splits=5,
    top_n_features=10,
    return_fold_details=True,
    verbose=False
):
    """
    Run GroupKFold cross-validation with feature selection inside each fold.

    This function is intentionally verbose in its returned JSON structure.
    It stores fold-level diagnostics so that equal AUC values observed with
    different correlation-pruning thresholds can be audited.

    Pipeline inside each fold:

    1. split train/test using GroupKFold
    2. scale X_train for LASSO selection
    3. run LASSO selection only on the training fold
    4. keep the top_n_features selected features
    5. scale selected train/test matrices
    6. fit LogisticRegression on the training fold
    7. predict probabilities on the test fold
    8. compute fold AUC

    Important:
    The test fold is never used for feature selection or scaling fit.
    """

    outer_cv = GroupKFold(
        n_splits=n_splits
    )

    aucs = []
    selected_counter = {}
    fold_details = []
    all_predictions = []

    fold_id = 0

    for train_idx, test_idx in outer_cv.split(
        X,
        y,
        groups
    ):

        fold_id += 1

        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        groups_train = groups.iloc[train_idx]
        groups_test = groups.iloc[test_idx]

        # ----------------------------------------------------
        # Feature selection is performed only on the train fold
        # ----------------------------------------------------

        selection_scaler = StandardScaler()

        X_train_scaled = pd.DataFrame(
            selection_scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )

        selected = lasso_selection(
            X_train_scaled,
            y_train
        )

        features = [
            item["feature"]
            for item in selected[:top_n_features]
        ]

        if len(features) == 0:
            raise ValueError(
                "LASSO selected zero features in fold "
                f"{fold_id}. Lower regularization or inspect the dataset."
            )

        for feature in features:
            selected_counter[feature] = (
                selected_counter.get(feature, 0) + 1
            )

        # ----------------------------------------------------
        # Final fold model on selected features
        # ----------------------------------------------------

        X_train_final = X_train[features]
        X_test_final = X_test[features]

        final_scaler = StandardScaler()

        X_train_final_scaled = final_scaler.fit_transform(
            X_train_final
        )

        X_test_final_scaled = final_scaler.transform(
            X_test_final
        )

        model = LogisticRegression(
            l1_ratio=1.0,
            solver="liblinear",
            C=1.0,
            max_iter=5000
        )

        model.fit(
            X_train_final_scaled,
            y_train
        )

        probabilities = model.predict_proba(
            X_test_final_scaled
        )[:, 1]

        # ----------------------------------------------------
        # Fold AUC
        # ----------------------------------------------------

        if len(np.unique(y_test)) < 2:
            fold_auc = None
        else:
            fold_auc = float(
                roc_auc_score(
                    y_test,
                    probabilities
                )
            )
            aucs.append(fold_auc)

        # ----------------------------------------------------
        # Prediction-level diagnostics
        # ----------------------------------------------------

        fold_predictions = []

        for row_index, patient, truth, probability in zip(
            X_test.index,
            groups_test,
            y_test,
            probabilities
        ):

            prediction_record = {
                "fold": int(fold_id),
                "row_index": _json_scalar(row_index),
                "patient": str(patient),
                "y_true": int(truth),
                "probability": float(probability),
                "predicted_class_0_5": int(probability >= 0.5)
            }

            fold_predictions.append(prediction_record)
            all_predictions.append(prediction_record)

        # ----------------------------------------------------
        # Model coefficient diagnostics
        # ----------------------------------------------------

        fold_model_coefficients = []

        for feature, coef in zip(
            features,
            model.coef_[0]
        ):
            fold_model_coefficients.append({
                "feature": feature,
                "coef": float(coef)
            })

        # ----------------------------------------------------
        # Fold detail object
        # ----------------------------------------------------

        fold_detail = {
            "fold": int(fold_id),
            "auc": fold_auc,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "train_indices": [int(i) for i in train_idx],
            "test_indices": [int(i) for i in test_idx],
            "train_patients": [str(x) for x in groups_train.tolist()],
            "test_patients": [str(x) for x in groups_test.tolist()],
            "y_train_counts": _value_counts_as_dict(y_train),
            "y_test_counts": _value_counts_as_dict(y_test),
            "selected_features": features,
            "selected_features_count": int(len(features)),
            "lasso_selected_features_full": [
                {
                    "feature": item["feature"],
                    "coef": float(item["coef"])
                }
                for item in selected
            ],
            "final_model_intercept": float(model.intercept_[0]),
            "final_model_coefficients": fold_model_coefficients,
            "predictions": fold_predictions,
            "predictions_sorted_by_probability": sorted(
                fold_predictions,
                key=lambda x: x["probability"],
                reverse=True
            )
        }

        if return_fold_details:
            fold_details.append(fold_detail)

        if verbose:
            auc_text = "NA" if fold_auc is None else f"{fold_auc:.3f}"
            print(
                f"Nested CV fold {fold_id}/{n_splits} | "
                f"AUC={auc_text} | "
                f"test patients={len(test_idx)} | "
                f"features={', '.join(features[:5])}"
            )

    if len(aucs) == 0:
        raise ValueError(
            "No valid fold AUC could be computed. "
            "Check class balance inside GroupKFold test folds."
        )

    result = {
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
        "folds": [float(x) for x in aucs],
        "stable_features": sorted(
            selected_counter.items(),
            key=lambda x: x[1],
            reverse=True
        ),
        "n_splits": int(n_splits),
        "top_n_features_per_fold": int(top_n_features),
        "all_predictions": all_predictions,
        "debug_note": (
            "fold_details contains train/test patients, selected features, "
            "fold coefficients and patient-level predicted probabilities. "
            "Use it to verify whether identical AUC values across pruning "
            "thresholds come from identical rankings or from different models "
            "with the same fold-wise ordering."
        )
    }

    if return_fold_details:
        result["fold_details"] = fold_details

    return result
