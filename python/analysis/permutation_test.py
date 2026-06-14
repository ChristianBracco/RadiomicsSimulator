import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from analysis.univariate_selection import select_features, normalize_selection_method


def _make_final_model():
    return LogisticRegression(
        solver="liblinear",
        l1_ratio=0.0,
        C=1.0,
        max_iter=5000,
    )


def single_nested_auc(
    X,
    y,
    groups,
    top_n_features=1,
    selection_method="lasso",
):
    """One grouped AUC estimate used inside the permutation test."""
    top_n_features = int(top_n_features)
    selection_method = normalize_selection_method(selection_method)

    if top_n_features < 1:
        raise ValueError("top_n_features must be >= 1.")

    outer_cv = GroupKFold(n_splits=5)
    predictions = []
    truths = []
    intercept_only_folds = 0

    for train_idx, test_idx in outer_cv.split(X, y, groups):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        selected = select_features(X_train, y_train, method=selection_method)
        features = [x["feature"] for x in selected[:top_n_features]]

        if len(features) == 0:
            intercept_only_folds += 1
            prob = np.full(shape=len(y_test), fill_value=float(np.mean(y_train)), dtype=float)
        else:
            X_train_final = X_train[features]
            X_test_final = X_test[features]

            scaler = StandardScaler()
            X_train_final = scaler.fit_transform(X_train_final)
            X_test_final = scaler.transform(X_test_final)

            model = _make_final_model()
            model.fit(X_train_final, y_train)
            prob = model.predict_proba(X_test_final)[:, 1]

        predictions.extend(prob)
        truths.extend(y_test)

    return roc_auc_score(truths, predictions), intercept_only_folds


def permutation_test(
    X,
    y,
    groups,
    observed_auc,
    n_permutations=1000,
    top_n_features=1,
    selection_method="lasso",
):
    """Permutation test with configurable feature-selection method."""
    selection_method = normalize_selection_method(selection_method)
    random_aucs = []
    intercept_only_total = 0

    for i in range(n_permutations):
        shuffled_y = pd.Series(np.random.permutation(y.values), index=y.index)

        auc, intercept_only_folds = single_nested_auc(
            X,
            shuffled_y,
            groups,
            top_n_features=top_n_features,
            selection_method=selection_method,
        )

        random_aucs.append(auc)
        intercept_only_total += int(intercept_only_folds)

        if (i + 1) % 50 == 0:
            print(f"Permutation {i+1}/{n_permutations}")

    random_aucs = np.array(random_aucs)
    p_value = np.mean(random_aucs >= observed_auc)

    return {
        "observed_auc": float(observed_auc),
        "mean_random_auc": float(random_aucs.mean()),
        "std_random_auc": float(random_aucs.std()),
        "max_random_auc": float(random_aucs.max()),
        "p_value": float(p_value),
        "n_permutations": int(n_permutations),
        "top_n_features": int(top_n_features),
        "selection_method": selection_method,
        "intercept_only_folds_total": int(intercept_only_total),
    }
