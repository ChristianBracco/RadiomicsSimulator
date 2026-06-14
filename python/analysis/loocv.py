import numpy as np

from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix

from analysis.univariate_selection import select_features, normalize_selection_method


def _make_final_model():
    return LogisticRegression(
        solver="liblinear",
        l1_ratio=0.0,
        C=1.0,
        max_iter=5000,
    )


def run_loocv(
    X,
    y,
    top_n_features=1,
    selection_method="lasso",
):
    """Leave-one-out cross-validation with configurable feature selection."""
    top_n_features = int(top_n_features)
    selection_method = normalize_selection_method(selection_method)

    if top_n_features < 1:
        raise ValueError("top_n_features must be >= 1.")

    loo = LeaveOneOut()
    predictions = []
    truths = []
    selected_counter = {}
    intercept_only_folds = 0
    iteration = 0

    for train_idx, test_idx in loo.split(X):
        iteration += 1

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        selected = select_features(X_train, y_train, method=selection_method)
        features = [x["feature"] for x in selected[:top_n_features]]

        for f in features:
            selected_counter[f] = selected_counter.get(f, 0) + 1

        if len(features) == 0:
            intercept_only_folds += 1
            prob = float(np.mean(y_train))
        else:
            X_train_final = X_train[features]
            X_test_final = X_test[features]

            scaler = StandardScaler()
            X_train_final = scaler.fit_transform(X_train_final)
            X_test_final = scaler.transform(X_test_final)

            model = _make_final_model()
            model.fit(X_train_final, y_train)
            prob = float(model.predict_proba(X_test_final)[0, 1])

        predictions.append(prob)
        truths.append(int(y_test.iloc[0]))

        if iteration % 5 == 0:
            print(f"LOOCV {iteration}/{len(X)}")

    auc = roc_auc_score(truths, predictions)
    predicted_class = [1 if p >= 0.5 else 0 for p in predictions]
    tn, fp, fn, tp = confusion_matrix(truths, predicted_class, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else None
    specificity = tn / (tn + fp) if (tn + fp) > 0 else None

    stable_features = sorted(selected_counter.items(), key=lambda x: x[1], reverse=True)

    return {
        "auc": float(auc),
        "sensitivity": None if sensitivity is None else float(sensitivity),
        "specificity": None if specificity is None else float(specificity),
        "stable_features": stable_features[:20],
        "top_n_features": int(top_n_features),
        "selection_method": selection_method,
        "intercept_only_folds": int(intercept_only_folds),
        "n_predictions": int(len(predictions)),
    }
