import numpy as np

from sklearn.model_selection import GroupKFold

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

def run_cv(
    X,
    y,
    groups
):

    gkf = GroupKFold(
        n_splits=5
    )

    aucs = []

    for train_idx, test_idx in gkf.split(
        X,
        y,
        groups
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        scaler = StandardScaler()

        X_train = scaler.fit_transform(
            X_train
        )

        X_test = scaler.transform(
            X_test
        )

        model = LogisticRegression(
            solver="liblinear",
            l1_ratio=1.0,
            C=1.0,
            max_iter=5000
        )

        model.fit(
            X_train,
            y_train
        )

        prob = model.predict_proba(
            X_test
        )[:,1]

        auc = roc_auc_score(
            y_test,
            prob
        )

        aucs.append(auc)

    return {

        "mean_auc":
        float(
            np.mean(aucs)
        ),

        "std_auc":
        float(
            np.std(aucs)
        ),

        "folds":
        [float(x) for x in aucs]
    }