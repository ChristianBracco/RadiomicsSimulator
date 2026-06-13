import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix


# ============================================================
# UTILITY
# ============================================================

def _make_logistic_model():
    """
    Candidate models: LogisticRegression NON LASSO.

    Nota:
    i modelli a 1/3/5 feature valutano feature già scelte
    da bootstrap-stability o da top-k LASSO del run corrente.
    Quindi qui non si fa ulteriore selezione L1.

    class_weight='balanced' aiuta quando le classi sono sbilanciate.
    """
    return LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        C=1.0,
        max_iter=5000
    )


def _safe_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(np.unique(y_true)) < 2:
        return None

    return float(
        roc_auc_score(
            y_true,
            y_prob
        )
    )


def _safe_confusion(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    sensitivity = None
    specificity = None

    if (tp + fn) > 0:
        sensitivity = float(tp / (tp + fn))

    if (tn + fp) > 0:
        specificity = float(tn / (tn + fp))

    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity": sensitivity,
        "specificity": specificity
    }


def _filter_existing(features, columns):
    return [
        f for f in features
        if f in columns
    ]


def _top_bootstrap_features(bootstrap, columns, n):
    """
    Prende le prime n feature bootstrap-stable realmente presenti
    in X_pruned.
    """
    if not bootstrap:
        return []

    out = []

    for item in bootstrap:
        f = item.get("feature")
        if f in columns and f not in out:
            out.append(f)

        if len(out) >= n:
            break

    return out


def _top_lasso_features(selected_features, columns, n):
    """
    Prende le prime n feature LASSO realmente presenti in X_pruned.
    """
    if not selected_features:
        return []

    out = []

    for item in selected_features:
        f = item.get("feature")
        if f in columns and f not in out:
            out.append(f)

        if len(out) >= n:
            break

    return out


def _candidate_feature_sets(X, selected_features=None, bootstrap=None):
    """
    Costruisce i modelli candidati.

    Regola:
    - 1 feature = prima bootstrap-stable;
    - 3 feature = prime 3 bootstrap-stable;
    - 5 feature = prime 5 bootstrap-stable;
    - top3/top5/top10 LASSO = confronto esplorativo.

    Se il bootstrap non ha abbastanza feature, usa fallback LASSO.
    """
    columns = list(X.columns)

    specs = [
        {
            "model_id": "one_feature_bootstrap",
            "label": "1 feature - top bootstrap-stable",
            "family": "bootstrap_stable",
            "requested_n": 1,
            "description": "Modello minimale costruito dalla feature più stabile al bootstrap."
        },
        {
            "model_id": "three_feature_bootstrap",
            "label": "3 feature - top bootstrap-stable",
            "family": "bootstrap_stable",
            "requested_n": 3,
            "description": "Modello ristretto costruito dalle prime 3 feature bootstrap-stable."
        },
        {
            "model_id": "five_feature_bootstrap",
            "label": "5 feature - top bootstrap-stable",
            "family": "bootstrap_stable",
            "requested_n": 5,
            "description": "Modello compatto costruito dalle prime 5 feature bootstrap-stable."
        }
    ]

    candidates = []

    for spec in specs:
        features = _top_bootstrap_features(
            bootstrap,
            columns,
            spec["requested_n"]
        )

        source = "top_bootstrap_stable"

        if len(features) < spec["requested_n"]:
            features = _top_lasso_features(
                selected_features,
                columns,
                spec["requested_n"]
            )
            source = "top_lasso_fallback"

        candidates.append({
            "model_id": spec["model_id"],
            "label": spec["label"],
            "family": spec["family"],
            "selection_source": source,
            "features": features,
            "n_features": len(features),
            "requested_n": spec["requested_n"],
            "description": spec["description"]
        })

    for n in [3, 5, 10]:
        features = _top_lasso_features(
            selected_features,
            columns,
            n
        )

        candidates.append({
            "model_id": f"lasso_top{n}_current",
            "label": f"{n} feature - top LASSO current run",
            "family": "current_run_lasso_topk",
            "selection_source": "top_lasso_current_run",
            "features": features,
            "n_features": len(features),
            "requested_n": n,
            "description": (
                f"Modello a {n} feature prese dalla selezione LASSO "
                "descrittiva del run corrente. Esplorativo, perché "
                "la selezione top-k deriva dal dataset completo."
            )
        })

    return candidates


# ============================================================
# VALUTAZIONE
# ============================================================

def _group_cv_evaluate(X, y, groups, features, n_splits=5):
    gkf = GroupKFold(
        n_splits=n_splits
    )

    y = pd.Series(y).reset_index(drop=True)
    groups = pd.Series(groups).reset_index(drop=True)
    X = X.reset_index(drop=True)

    fold_aucs = []
    all_predictions = []
    fold_details = []

    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(X, y, groups),
        start=1
    ):
        X_train = X.iloc[train_idx][features]
        X_test = X.iloc[test_idx][features]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(
            X_train
        )

        X_test_scaled = scaler.transform(
            X_test
        )

        model = _make_logistic_model()

        model.fit(
            X_train_scaled,
            y_train
        )

        prob = model.predict_proba(
            X_test_scaled
        )[:, 1]

        fold_auc = _safe_auc(
            y_test,
            prob
        )

        if fold_auc is not None:
            fold_aucs.append(
                fold_auc
            )

        patient_ids = [
            str(x) for x in groups.iloc[test_idx].tolist()
        ]

        preds = []

        for patient, yt, p in zip(
            patient_ids,
            y_test.tolist(),
            prob.tolist()
        ):
            row = {
                "fold": int(fold),
                "patient": str(patient),
                "true_label": int(yt),
                "predicted_probability": float(p),
                "predicted_class_05": int(p >= 0.5)
            }

            preds.append(row)
            all_predictions.append(row)

        coefficients = [
            {
                "feature": f,
                "coef": float(c)
            }
            for f, c in zip(
                features,
                model.coef_[0]
            )
        ]

        fold_details.append({
            "fold": int(fold),
            "auc": fold_auc,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "test_patients": patient_ids,
            "test_responders": int((y_test == 1).sum()),
            "test_non_responders": int((y_test == 0).sum()),
            "coefficients": coefficients,
            "intercept": float(model.intercept_[0]),
            "predictions": preds
        })

    pooled_auc = _safe_auc(
        [x["true_label"] for x in all_predictions],
        [x["predicted_probability"] for x in all_predictions]
    )

    cm = _safe_confusion(
        [x["true_label"] for x in all_predictions],
        [x["predicted_probability"] for x in all_predictions]
    )

    return {
        "mean_auc": float(np.mean(fold_aucs)) if fold_aucs else None,
        "std_auc": float(np.std(fold_aucs)) if fold_aucs else None,
        "folds": [float(x) for x in fold_aucs],
        "pooled_auc": pooled_auc,
        "sensitivity": cm["sensitivity"],
        "specificity": cm["specificity"],
        "confusion_matrix": {
            "tn": cm["tn"],
            "fp": cm["fp"],
            "fn": cm["fn"],
            "tp": cm["tp"]
        },
        "all_predictions": all_predictions,
        "fold_details": fold_details
    }


def _loocv_evaluate(X, y, features):
    loo = LeaveOneOut()

    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)

    predictions = []

    for train_idx, test_idx in loo.split(X):
        X_train = X.iloc[train_idx][features]
        X_test = X.iloc[test_idx][features]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(
            X_train
        )

        X_test_scaled = scaler.transform(
            X_test
        )

        model = _make_logistic_model()

        model.fit(
            X_train_scaled,
            y_train
        )

        prob = float(
            model.predict_proba(
                X_test_scaled
            )[0, 1]
        )

        predictions.append({
            "row_index": int(test_idx[0]),
            "true_label": int(y_test.iloc[0]),
            "predicted_probability": prob,
            "predicted_class_05": int(prob >= 0.5)
        })

    auc = _safe_auc(
        [x["true_label"] for x in predictions],
        [x["predicted_probability"] for x in predictions]
    )

    cm = _safe_confusion(
        [x["true_label"] for x in predictions],
        [x["predicted_probability"] for x in predictions]
    )

    return {
        "auc": auc,
        "sensitivity": cm["sensitivity"],
        "specificity": cm["specificity"],
        "confusion_matrix": {
            "tn": cm["tn"],
            "fp": cm["fp"],
            "fn": cm["fn"],
            "tp": cm["tp"]
        },
        "predictions": predictions
    }


def _fit_full_model(X, y, features):
    X_final = X[features]

    scaler = StandardScaler()

    Xs = scaler.fit_transform(
        X_final
    )

    model = _make_logistic_model()

    model.fit(
        Xs,
        y
    )

    prob = model.predict_proba(
        Xs
    )[:, 1]

    auc = _safe_auc(
        y,
        prob
    )

    coefficients = [
        {
            "feature": f,
            "coef": float(c)
        }
        for f, c in zip(
            features,
            model.coef_[0]
        )
    ]

    return {
        "apparent_training_auc": auc,
        "intercept": float(model.intercept_[0]),
        "coefficients": coefficients
    }


def _flatten_model(item):
    """
    Aggiunge campi flat richiesti dal tuo run_analysis.py:

        model['group_cv_mean_auc']
        model['loocv_auc']
        ...

    Mantiene anche la struttura nested completa.
    """
    group_cv = item.get(
        "group_cv",
        {}
    ) or {}

    loocv = item.get(
        "loocv",
        {}
    ) or {}

    full = item.get(
        "full_model",
        {}
    ) or {}

    item["group_cv_mean_auc"] = group_cv.get(
        "mean_auc"
    )

    item["group_cv_std_auc"] = group_cv.get(
        "std_auc"
    )

    item["group_cv_pooled_auc"] = group_cv.get(
        "pooled_auc"
    )

    item["group_cv_sensitivity"] = group_cv.get(
        "sensitivity"
    )

    item["group_cv_specificity"] = group_cv.get(
        "specificity"
    )

    item["loocv_auc"] = loocv.get(
        "auc"
    )

    item["loocv_sensitivity"] = loocv.get(
        "sensitivity"
    )

    item["loocv_specificity"] = loocv.get(
        "specificity"
    )

    item["apparent_training_auc"] = full.get(
        "apparent_training_auc"
    )

    return item


def benchmark_candidate_models(
    X,
    y,
    groups,
    selected_features=None,
    bootstrap=None,
    pruning_threshold=None
):
    candidates = _candidate_feature_sets(
        X,
        selected_features=selected_features,
        bootstrap=bootstrap
    )

    results = []

    for spec in candidates:
        features = spec.get(
            "features",
            []
        )

        if len(features) == 0:
            item = {
                **spec,
                "status": "skipped",
                "skip_reason": (
                    "No usable features available after pruning, "
                    "bootstrap fallback or LASSO fallback."
                )
            }

            results.append(
                _flatten_model(item)
            )
            continue

        group_cv = _group_cv_evaluate(
            X,
            y,
            groups,
            features
        )

        loocv = _loocv_evaluate(
            X,
            y,
            features
        )

        full_model = _fit_full_model(
            X,
            y,
            features
        )

        item = {
            **spec,
            "status": "ok",
            "group_cv": group_cv,
            "loocv": loocv,
            "full_model": full_model
        }

        results.append(
            _flatten_model(item)
        )

    ok_models = [
        m for m in results
        if m.get("status") == "ok"
    ]

    def _score(m):
        group_auc = m.get(
            "group_cv_mean_auc"
        ) or 0

        loo_auc = m.get(
            "loocv_auc"
        ) or 0

        n_features = m.get(
            "n_features",
            99
        )

        return (
            (group_auc + loo_auc) / 2
            - max(0, n_features - 1) * 0.002
        )

    best = None

    if ok_models:
        best = sorted(
            ok_models,
            key=_score,
            reverse=True
        )[0]["model_id"]

    return {
        "purpose": (
            "Compare bootstrap-stable fixed-size models against current "
            "LASSO top-k candidates."
        ),
        "important_caution": (
            "Bootstrap-stable 1/3/5 feature models are preferred for "
            "interpretability. LASSO top-k models remain exploratory because "
            "the top-k set is selected from the current full run."
        ),
        "pruning_threshold": pruning_threshold,
        "models": results,
        "summary": results,
        "best_model_id_by_internal_score": best,
        "n_models": len(results),
        "n_ok_models": len(ok_models)
    }


# ============================================================
# ALIAS COMPATIBILI
# ============================================================

def compare_candidate_models(
    X,
    y,
    groups,
    selected_features_detail=None,
    bootstrap_features_detail=None,
    n_splits=5,
    return_details=True,
    pruning_threshold=None,
    **kwargs
):
    """
    Compatibile con run_analysis.py che usa:

        candidate_models = compare_candidate_models(...)
        for model in candidate_models["summary"]:
            print(model["group_cv_mean_auc"])

    Quindi restituisce un dizionario con:
    - summary: lista modelli con campi flat;
    - models: stessa lista per dashboard.
    """
    if pruning_threshold is None:
        pruning_threshold = kwargs.get(
            "threshold",
            None
        )

    return benchmark_candidate_models(
        X=X,
        y=y,
        groups=groups,
        selected_features=selected_features_detail,
        bootstrap=bootstrap_features_detail,
        pruning_threshold=pruning_threshold
    )


def run_candidate_model_benchmark(
    X,
    y,
    groups,
    selected_features=None,
    bootstrap=None,
    pruning_threshold=None,
    **kwargs
):
    return benchmark_candidate_models(
        X=X,
        y=y,
        groups=groups,
        selected_features=selected_features,
        bootstrap=bootstrap,
        pruning_threshold=pruning_threshold
    )
