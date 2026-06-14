from analysis.univariate_selection import select_features


def lasso_selection(X, y):
    """Backward-compatible wrapper for true L1 logistic feature selection."""
    return select_features(X, y, method="lasso")
