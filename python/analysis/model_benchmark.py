import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix

from analysis.univariate_selection import select_features, normalize_selection_method, selection_method_label


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
        l1_ratio=0.0,
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


def _top_method_features(X, y, method, n):
    """Return the first n features ranked by the requested selector."""
    ranked = select_features(X, y, method=method)
    columns = set(X.columns)
    out = []
    for item in ranked:
        f = item.get("feature")
        if f in columns and f not in out:
            out.append(f)
        if len(out) >= n:
            break
    return out, ranked[:max(n, 10)]


def _candidate_feature_sets(
    X,
    y=None,
    selected_features=None,
    bootstrap=None,
    selection_method="lasso",
    selection_methods_to_compare=None,
):
    """
    Build candidate models.

    Families:
    - bootstrap-stable 1/3/5 features;
    - current primary selector top 1/3/5;
    - univariate/penalized selector comparison: Pearson, Spearman, AUC,
      Mann-Whitney, mutual information and LASSO.
    """
    columns = list(X.columns)
    selection_method = normalize_selection_method(selection_method)

    if selection_methods_to_compare is None:
        selection_methods_to_compare = [
            "pearson",
            "spearman",
            "auc",
            "mannwhitney",
            "mutual_info",
            "lasso",
        ]

    # Normalize and de-duplicate while preserving order.
    normalized_methods = []
    for method in selection_methods_to_compare:
        try:
            m = normalize_selection_method(method)
        except Exception:
            continue
        if m not in normalized_methods:
            normalized_methods.append(m)

    candidates = []

    # ------------------------------------------------------------
    # Bootstrap-stable models
    # ------------------------------------------------------------
    bootstrap_specs = [
        ("one_feature_bootstrap", "1 feature - top bootstrap-stable", 1),
        ("three_feature_bootstrap", "3 feature - top bootstrap-stable", 3),
        ("five_feature_bootstrap", "5 feature - top bootstrap-stable", 5),
    ]

    for model_id, label, n in bootstrap_specs:
        features = _top_bootstrap_features(bootstrap, columns, n)
        source = "top_bootstrap_stable"

        if len(features) < n:
            features = _top_lasso_features(selected_features, columns, n)
            source = "top_primary_selection_fallback"

        candidates.append({
            "model_id": model_id,
            "label": label,
            "family": "bootstrap_stable",
            "selection_method": selection_method,
            "selection_source": source,
            "features": features,
            "n_features": len(features),
            "requested_n": n,
            "description": (
                "Modello compatto costruito dalle feature più stabili al bootstrap. "
                "Se non disponibili, usa fallback dalla selezione primaria."
            ),
        })

    # ------------------------------------------------------------
    # Primary selector top-k models
    # ------------------------------------------------------------
    for n in [1, 3, 5]:
        features = _top_lasso_features(selected_features, columns, n)
        candidates.append({
            "model_id": f"primary_{selection_method}_top{n}",
            "label": f"{n} feature - primary {selection_method_label(selection_method)}",
            "family": "primary_selector_topk",
            "selection_method": selection_method,
            "selection_source": f"top_{selection_method}_current_run",
            "features": features,
            "n_features": len(features),
            "requested_n": n,
            "description": (
                f"Modello a {n} feature prese dalla selezione primaria corrente "
                f"({selection_method_label(selection_method)}). Esplorativo se la "
                "selezione deriva dal dataset completo."
            ),
        })

    # ------------------------------------------------------------
    # Head-to-head selector comparison
    # ------------------------------------------------------------
    if y is not None:
        for method in normalized_methods:
            for n in [1, 3, 5]:
                features, ranking = _top_method_features(X, y, method, n)
                candidates.append({
                    "model_id": f"{method}_top{n}",
                    "label": f"{n} feature - {selection_method_label(method)}",
                    "family": "selector_comparison",
                    "selection_method": method,
                    "selection_source": f"top_{method}_univariate_or_penalized",
                    "features": features,
                    "n_features": len(features),
                    "requested_n": n,
                    "selection_ranking_preview": ranking,
                    "description": (
                        "Confronto head-to-head: cambia il metodo di selezione, "
                        "ma il classificatore finale resta LogisticRegression."
                    ),
                })

    # De-duplicate identical model_id entries while preserving order.
    out = []
    seen = set()
    for item in candidates:
        mid = item.get("model_id")
        if mid in seen:
            continue
        seen.add(mid)
        out.append(item)

    return out

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
    pruning_threshold=None,
    selection_method="lasso",
    selection_methods_to_compare=None,
):
    selection_method = normalize_selection_method(selection_method)
    candidates = _candidate_feature_sets(
        X,
        y=y,
        selected_features=selected_features,
        bootstrap=bootstrap,
        selection_method=selection_method,
        selection_methods_to_compare=selection_methods_to_compare,
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
                    "bootstrap fallback or selector fallback."
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
            "Compare bootstrap-stable fixed-size models, the primary selector, "
            "and univariate/penalized feature-selection strategies."
        ),
        "important_caution": (
            "Bootstrap-stable and 1-feature models are preferred for "
            "interpretability. Selector-comparison models are sensitivity "
            "analyses because their top-k set is ranked on the current full run."
        ),
        "pruning_threshold": pruning_threshold,
        "selection_method": selection_method,
        "selection_methods_compared": selection_methods_to_compare,
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
    selection_method="lasso",
    selection_methods_to_compare=None,
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

    if selection_method is None:
        selection_method = kwargs.get("feature_selection_method", "lasso")
    if selection_methods_to_compare is None:
        selection_methods_to_compare = kwargs.get("selection_methods_to_compare", None)

    return benchmark_candidate_models(
        X=X,
        y=y,
        groups=groups,
        selected_features=selected_features_detail,
        bootstrap=bootstrap_features_detail,
        pruning_threshold=pruning_threshold,
        selection_method=selection_method,
        selection_methods_to_compare=selection_methods_to_compare,
    )


def run_candidate_model_benchmark(
    X,
    y,
    groups,
    selected_features=None,
    bootstrap=None,
    pruning_threshold=None,
    selection_method="lasso",
    selection_methods_to_compare=None,
    **kwargs
):
    return benchmark_candidate_models(
        X=X,
        y=y,
        groups=groups,
        selected_features=selected_features,
        bootstrap=bootstrap,
        pruning_threshold=pruning_threshold,
        selection_method=selection_method,
        selection_methods_to_compare=selection_methods_to_compare,
    )
