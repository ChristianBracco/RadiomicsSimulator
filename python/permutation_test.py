import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from analysis.lasso_selection import lasso_selection


def single_nested_auc(
    X,
    y,
    groups,
    top_n_features=1,
):
    """
    One nested/grouped AUC estimate used inside the permutation test.
    Feature selection is performed inside each training fold.
    """

    top_n_features = int(
        top_n_features
    )

    if top_n_features < 1:
        raise ValueError(
            "top_n_features must be >= 1."
        )

    outer_cv = GroupKFold(
        n_splits=5
    )

    predictions = []
    truths = []

    for train_idx, test_idx in outer_cv.split(
        X,
        y,
        groups
    ):

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

        # During permutation testing the labels are deliberately randomized.
        # In some random permutations, especially after aggressive pruning
        # and with a 1-feature primary model, LASSO can legitimately select
        # zero features. This must not crash the whole analysis.
        # Statistically, the correct fallback is an intercept-only model:
        # all test patients receive the training-fold event prevalence.
        if len(features) == 0:
            prob = np.full(
                shape=len(y_test),
                fill_value=float(np.mean(y_train)),
                dtype=float
            )
        else:
            X_train_final = X_train[features]
            X_test_final = X_test[features]

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
            )[:, 1]

        predictions.extend(prob)
        truths.extend(y_test)

    return roc_auc_score(
        truths,
        predictions
    )


def permutation_test(
    X,
    y,
    groups,
    observed_auc,
    n_permutations=1000,
    top_n_features=1,
):
    """
    Permutation test with configurable number of selected features.

    Robustness note:
    Permuted labels can lead LASSO to select zero features in some folds.
    Those folds are treated as intercept-only models instead of raising an
    exception, so the backend can complete and the dashboard does not show
    stale results from a previous run.
    """

    random_aucs = []

    for i in range(
        n_permutations
    ):

        shuffled_y = pd.Series(
            np.random.permutation(y.values),
            index=y.index
        )

        auc = single_nested_auc(
            X,
            shuffled_y,
            groups,
            top_n_features=top_n_features
        )

        random_aucs.append(
            auc
        )

        if (i + 1) % 50 == 0:
            print(
                f"Permutation {i+1}/{n_permutations}"
            )

    random_aucs = np.array(
        random_aucs
    )

    p_value = np.mean(
        random_aucs >= observed_auc
    )

    return {
        "observed_auc": float(observed_auc),
        "mean_random_auc": float(
            random_aucs.mean()
        ),
        "std_random_auc": float(
            random_aucs.std()
        ),
        "max_random_auc": float(
            random_aucs.max()
        ),
        "p_value": float(p_value),
        "n_permutations": int(n_permutations),
        "top_n_features": int(top_n_features)
    }
