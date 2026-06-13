import numpy as np
import pandas as pd

from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix
)

from analysis.lasso_selection import (
    lasso_selection
)


def run_loocv(
    X,
    y,
    top_n_features=1,
):
    """
    Leave-one-out cross-validation with LASSO feature selection inside each
    training split.

    top_n_features is now explicit. Use top_n_features=1 as the primary
    compact-signature analysis; 3/5/10 can be used as exploratory stress tests.
    """

    top_n_features = int(
        top_n_features
    )

    if top_n_features < 1:
        raise ValueError(
            "top_n_features must be >= 1."
        )

    loo = LeaveOneOut()

    predictions = []
    truths = []
    selected_counter = {}

    iteration = 0

    for train_idx, test_idx in loo.split(X):

        iteration += 1

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        scaler = StandardScaler()

        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )

        selected = lasso_selection(
            X_train_scaled,
            y_train
        )

        features = [
            x["feature"]
            for x in selected[:top_n_features]
        ]

        if len(features) == 0:
            raise ValueError(
                "LASSO selected zero features in LOOCV. "
                "Lower regularization or inspect the dataset."
            )

        for f in features:
            selected_counter[f] = (
                selected_counter.get(
                    f,
                    0
                ) + 1
            )

        X_train_final = X_train[
            features
        ]

        X_test_final = X_test[
            features
        ]

        scaler2 = StandardScaler()

        X_train_final = scaler2.fit_transform(
            X_train_final
        )

        X_test_final = scaler2.transform(
            X_test_final
        )

        model = LogisticRegression(
            l1_ratio=1.0,
            solver="liblinear",
            C=1.0,
            max_iter=5000
        )

        model.fit(
            X_train_final,
            y_train
        )

        prob = model.predict_proba(
            X_test_final
        )[0, 1]

        predictions.append(
            prob
        )

        truths.append(
            int(
                y_test.iloc[0]
            )
        )

        if iteration % 5 == 0:
            print(
                f"LOOCV {iteration}/{len(X)}"
            )

    auc = roc_auc_score(
        truths,
        predictions
    )

    predicted_class = [
        1 if p >= 0.5 else 0
        for p in predictions
    ]

    tn, fp, fn, tp = confusion_matrix(
        truths,
        predicted_class,
        labels=[0, 1]
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else None
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else None
    )

    stable_features = sorted(
        selected_counter.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return {
        "auc": float(auc),
        "sensitivity": None if sensitivity is None else float(sensitivity),
        "specificity": None if specificity is None else float(specificity),
        "stable_features": stable_features[:20],
        "top_n_features": int(top_n_features),
        "n_predictions": int(len(predictions))
    }
