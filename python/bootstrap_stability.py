import numpy as np
import pandas as pd

from analysis.lasso_selection import (
    lasso_selection
)

def bootstrap_stability(
    X,
    y,
    n_bootstrap=1000
):

    feature_counts = {}

    n = len(X)

    for i in range(
        n_bootstrap
    ):

        idx = np.random.choice(
            n,
            n,
            replace=True
        )

        X_boot = X.iloc[idx]
        y_boot = y.iloc[idx]

        selected = lasso_selection(
            X_boot,
            y_boot
        )

        for item in selected[:10]:

            feature = item["feature"]

            feature_counts[
                feature
            ] = feature_counts.get(
                feature,
                0
            ) + 1

        if (i + 1) % 100 == 0:

            print(
                f"Bootstrap {i+1}/{n_bootstrap}"
            )

    result = []

    for feature, count in feature_counts.items():

        result.append({

            "feature": feature,

            "frequency":
                count / n_bootstrap
        })

    result.sort(
        key=lambda x:
        x["frequency"],
        reverse=True
    )

    return result