import numpy as np

from analysis.univariate_selection import select_features, normalize_selection_method


def bootstrap_stability(
    X,
    y,
    n_bootstrap=1000,
    top_n_features=10,
    selection_method="lasso",
):
    """Bootstrap feature-selection stability for the chosen selector."""
    selection_method = normalize_selection_method(selection_method)
    top_n_features = int(top_n_features)

    feature_counts = {}
    n = len(X)

    for i in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        X_boot = X.iloc[idx]
        y_boot = y.iloc[idx]

        selected = select_features(X_boot, y_boot, method=selection_method)

        for item in selected[:top_n_features]:
            feature = item["feature"]
            feature_counts[feature] = feature_counts.get(feature, 0) + 1

        if (i + 1) % 100 == 0:
            print(f"Bootstrap {i+1}/{n_bootstrap}")

    result = []
    for feature, count in feature_counts.items():
        result.append({
            "feature": feature,
            "frequency": count / n_bootstrap,
            "selection_method": selection_method,
        })

    result.sort(key=lambda x: x["frequency"], reverse=True)
    return result
