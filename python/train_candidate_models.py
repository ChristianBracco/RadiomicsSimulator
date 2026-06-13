from pathlib import Path
import json

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib


def _safe_threshold(threshold):
    return str(threshold).replace(".", "")


def _default_models_dir():
    """
    train_candidate_models.py si trova in:
        server/server/python/train_candidate_models.py

    La cartella models deve essere:
        server/server/models
    """
    return (
        Path(__file__).resolve().parent.parent
        / "models"
    )


def _extract_candidate_items(candidate_comparison=None, candidate_models=None):
    """
    Accetta tutte le forme che abbiamo usato nelle patch precedenti:

    1) lista pura:
       [ {model1}, {model2}, ... ]

    2) dizionario con models:
       { "models": [...] }

    3) dizionario con summary:
       { "summary": [...] }

    Nel tuo run attuale candidate_models è verosimilmente un dizionario
    con summary/models.
    """
    source = candidate_models

    if source is None:
        source = candidate_comparison

    if source is None:
        return []

    if isinstance(source, list):
        return source

    if isinstance(source, dict):
        if isinstance(source.get("models"), list):
            return source["models"]

        if isinstance(source.get("summary"), list):
            return source["summary"]

        if isinstance(source.get("candidate_models"), list):
            return source["candidate_models"]

        if isinstance(source.get("results"), list):
            return source["results"]

    return []


def _make_pipeline():
    return Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                solver="liblinear",
                class_weight="balanced",
                C=1.0,
                max_iter=5000
            )
        )
    ])


def train_and_save_candidate_models(
    X=None,
    y=None,
    candidate_comparison=None,
    models_dir=None,
    threshold=0.85,
    X_pruned=None,
    candidate_models=None,
    output_dir=None,
    model_dir=None,
    dataset_summary=None,
    validation_summary=None,
    **kwargs
):
    """
    Versione robusta e retrocompatibile.

    Corregge l'errore:

        TypeError: train_and_save_candidate_models()
        got an unexpected keyword argument 'X_pruned'

    Perché accetta sia la firma nuova:

        train_and_save_candidate_models(
            X=...,
            y=...,
            candidate_comparison=...,
            models_dir=...
        )

    sia la firma usata dal tuo run_analysis.py:

        train_and_save_candidate_models(
            X_pruned=X_pruned,
            y=y,
            candidate_models=candidate_models,
            ...
        )

    Parametri extra come dataset_summary e validation_summary vengono
    salvati nei metadata ma non sono obbligatori.
    """
    if X is None:
        X = X_pruned

    if X is None:
        raise ValueError(
            "Missing X or X_pruned."
        )

    if y is None:
        y = kwargs.get(
            "target",
            None
        )

    if y is None:
        raise ValueError(
            "Missing y."
        )

    if models_dir is None:
        models_dir = model_dir

    if models_dir is None:
        models_dir = kwargs.get(
            "models_output_dir",
            None
        )

    if models_dir is None:
        models_dir = _default_models_dir()

    models_dir = Path(models_dir)
    models_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    candidate_items = _extract_candidate_items(
        candidate_comparison=candidate_comparison,
        candidate_models=candidate_models
    )

    saved = []
    skipped = []

    for item in candidate_items:
        if not isinstance(item, dict):
            skipped.append({
                "reason": "candidate item is not a dict",
                "item": str(item)
            })
            continue

        if item.get("status") != "ok":
            skipped.append({
                "model_id": item.get("model_id"),
                "label": item.get("label"),
                "status": item.get("status"),
                "reason": item.get("skip_reason")
            })
            continue

        features = item.get(
            "features",
            []
        )

        if not features:
            skipped.append({
                "model_id": item.get("model_id"),
                "label": item.get("label"),
                "reason": "empty feature list"
            })
            continue

        missing = [
            f for f in features
            if f not in X.columns
        ]

        if missing:
            skipped.append({
                "model_id": item.get("model_id"),
                "label": item.get("label"),
                "reason": "features missing in X",
                "missing_features": missing
            })
            continue

        model_id = item.get(
            "model_id",
            "candidate_model"
        )

        pipeline = _make_pipeline()

        pipeline.fit(
            X[features],
            y
        )

        payload = {
            "model_id": model_id,
            "label": item.get("label"),
            "family": item.get("family"),
            "selection_source": item.get("selection_source"),
            "threshold": threshold,
            "features": features,
            "pipeline": pipeline
        }

        joblib_path = (
            models_dir
            / f"theradiomics_threshold_{_safe_threshold(threshold)}_{model_id}.joblib"
        )

        joblib.dump(
            payload,
            joblib_path
        )

        saved_item = {
            "model_id": model_id,
            "label": item.get("label"),
            "family": item.get("family"),
            "selection_source": item.get("selection_source"),
            "features": features,
            "n_features": len(features),
            "joblib_path": str(joblib_path),
            "group_cv_mean_auc": item.get("group_cv_mean_auc"),
            "group_cv_std_auc": item.get("group_cv_std_auc"),
            "group_cv_pooled_auc": item.get("group_cv_pooled_auc"),
            "group_cv_sensitivity": item.get("group_cv_sensitivity"),
            "group_cv_specificity": item.get("group_cv_specificity"),
            "loocv_auc": item.get("loocv_auc"),
            "loocv_sensitivity": item.get("loocv_sensitivity"),
            "loocv_specificity": item.get("loocv_specificity"),
            "apparent_training_auc": item.get("apparent_training_auc")
        }

        saved.append(
            saved_item
        )

    metadata = {
        "threshold": threshold,
        "n_saved_models": len(saved),
        "n_skipped_models": len(skipped),
        "saved_models": saved,
        "skipped_models": skipped,
        "dataset_summary": dataset_summary,
        "validation_summary": validation_summary
    }

    metadata_path = (
        models_dir
        / f"theradiomics_threshold_{_safe_threshold(threshold)}_candidate_models_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2
        )

    saved_model_paths = [
        item.get("joblib_path")
        for item in saved
        if item.get("joblib_path")
    ]

    return {
        # Nome usato dal tuo run_analysis.py attuale
        "collection_metadata_path": str(metadata_path),

        # Alias retrocompatibili usati nelle patch precedenti
        "metadata_path": str(metadata_path),
        "candidate_models_metadata_path": str(metadata_path),

        # Alias richiesto dal tuo run_analysis.py attuale
        # che itera:
        #     for saved_model in saved_candidate_models["models"]:
        "models": saved,

        # Informazioni sui modelli salvati
        "saved_models": saved,
        "skipped_models": skipped,
        "saved_model_paths": saved_model_paths,
        "n_saved_models": len(saved),
        "n_skipped_models": len(skipped)
    }
