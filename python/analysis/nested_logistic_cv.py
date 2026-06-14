import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from analysis.univariate_selection import select_features, normalize_selection_method


# ============================================================
# JSON HELPERS
# ============================================================

def _json_scalar(value):
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
    counts = series.value_counts().to_dict()
    return {str(_json_scalar(k)): int(v) for k, v in counts.items()}


def _make_final_model():
    return LogisticRegression(
        solver="liblinear",
        l1_ratio=0.0,
        C=1.0,
        max_iter=5000,
    )


# ============================================================
# NESTED / FOLD-WISE VALIDATION WITH DEBUG DETAILS
# ============================================================

def run_nested_cv(
    X,
    y,
    groups,
    n_splits=5,
    top_n_features=10,
    selection_method="lasso",
    return_fold_details=True,
    verbose=False,
):
    """
    Run GroupKFold cross-validation with feature selection inside each fold.

    The feature-selection method is configurable. Supported values include:
    lasso, pearson, spearman, auc, mannwhitney, mutual_info.

    Important:
    The test fold is never used for feature selection or scaling fit.
    """
    selection_method = normalize_selection_method(selection_method)
    top_n_features = int(top_n_features)

    outer_cv = GroupKFold(n_splits=n_splits)

    aucs = []
    selected_counter = {}
    fold_details = []
    all_predictions = []
    intercept_only_folds = 0

    fold_id = 0

    for train_idx, test_idx in outer_cv.split(X, y, groups):
        fold_id += 1

        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        groups_train = groups.iloc[train_idx]
        groups_test = groups.iloc[test_idx]

        selected = select_features(
            X_train,
            y_train,
            method=selection_method,
        )

        features = [item["feature"] for item in selected[:top_n_features]]

        for feature in features:
            selected_counter[feature] = selected_counter.get(feature, 0) + 1

        fold_model_coefficients = []

        if len(features) == 0:
            intercept_only_folds += 1
            probabilities = np.full(
                shape=len(y_test),
                fill_value=float(np.mean(y_train)),
                dtype=float,
            )
            final_model_intercept = float(np.log(np.mean(y_train) / (1 - np.mean(y_train)))) if 0 < np.mean(y_train) < 1 else 0.0
        else:
            X_train_final = X_train[features]
            X_test_final = X_test[features]

            final_scaler = StandardScaler()
            X_train_final_scaled = final_scaler.fit_transform(X_train_final)
            X_test_final_scaled = final_scaler.transform(X_test_final)

            model = _make_final_model()
            model.fit(X_train_final_scaled, y_train)
            probabilities = model.predict_proba(X_test_final_scaled)[:, 1]
            final_model_intercept = float(model.intercept_[0])

            for feature, coef in zip(features, model.coef_[0]):
                fold_model_coefficients.append({
                    "feature": feature,
                    "coef": float(coef),
                })

        if len(np.unique(y_test)) < 2:
            fold_auc = None
        else:
            fold_auc = float(roc_auc_score(y_test, probabilities))
            aucs.append(fold_auc)

        fold_predictions = []
        for row_index, patient, truth, probability in zip(X_test.index, groups_test, y_test, probabilities):
            prediction_record = {
                "fold": int(fold_id),
                "row_index": _json_scalar(row_index),
                "patient": str(patient),
                "y_true": int(truth),
                "probability": float(probability),
                "predicted_class_0_5": int(probability >= 0.5),
            }
            fold_predictions.append(prediction_record)
            all_predictions.append(prediction_record)

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
            "selection_method": selection_method,
            "selected_features": features,
            "selected_features_count": int(len(features)),
            "intercept_only": bool(len(features) == 0),
            "selected_features_full": selected,
            # Backward-compatible alias used by older dashboard/debug code.
            "lasso_selected_features_full": selected,
            "final_model_intercept": final_model_intercept,
            "final_model_coefficients": fold_model_coefficients,
            "predictions": fold_predictions,
            "predictions_sorted_by_probability": sorted(
                fold_predictions,
                key=lambda x: x["probability"],
                reverse=True,
            ),
        }

        if return_fold_details:
            fold_details.append(fold_detail)

        if verbose:
            auc_text = "NA" if fold_auc is None else f"{fold_auc:.3f}"
            feat_text = ", ".join(features[:5]) if features else "intercept-only"
            print(
                f"Nested CV fold {fold_id}/{n_splits} | "
                f"AUC={auc_text} | "
                f"test patients={len(test_idx)} | "
                f"selection={selection_method} | "
                f"features={feat_text}"
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
        "stable_features": sorted(selected_counter.items(), key=lambda x: x[1], reverse=True),
        "n_splits": int(n_splits),
        "top_n_features_per_fold": int(top_n_features),
        "selection_method": selection_method,
        "intercept_only_folds": int(intercept_only_folds),
        "all_predictions": all_predictions,
        "debug_note": (
            "fold_details contains train/test patients, selected features, "
            "fold coefficients and patient-level predicted probabilities. "
            "Feature selection is performed inside each training fold."
        ),
    }

    if return_fold_details:
        result["fold_details"] = fold_details

    return result
