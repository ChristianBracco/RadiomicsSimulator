from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.feature_selection import mutual_info_classif
except Exception:  # pragma: no cover
    mutual_info_classif = None


VALID_SELECTION_METHODS = [
    "lasso",
    "pearson",
    "spearman",
    "auc",
    "mannwhitney",
    "mutual_info",
]


def normalize_selection_method(method: str | None) -> str:
    """Normalize aliases used by the dashboard/backend."""
    value = str(method or "lasso").strip().lower().replace("-", "_").replace(" ", "_")

    aliases = {
        "l1": "lasso",
        "logistic_l1": "lasso",
        "l1_logistic": "lasso",
        "true_lasso": "lasso",
        "point_biserial": "pearson",
        "pointbiserial": "pearson",
        "corr": "pearson",
        "correlation": "pearson",
        "univariate_auc": "auc",
        "roc_auc": "auc",
        "mw": "mannwhitney",
        "mann_whitney": "mannwhitney",
        "mann_whitney_u": "mannwhitney",
        "mwu": "mannwhitney",
        "mi": "mutual_info",
        "mutual_information": "mutual_info",
    }

    value = aliases.get(value, value)

    if value not in VALID_SELECTION_METHODS:
        raise ValueError(
            f"Unknown feature_selection_method={method!r}. "
            f"Allowed: {', '.join(VALID_SELECTION_METHODS)}"
        )

    return value


def _as_numeric_series(values) -> pd.Series:
    s = pd.Series(values)
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    return s


def _valid_xy(x, y):
    xs = _as_numeric_series(x)
    ys = _as_numeric_series(y)
    mask = xs.notna() & ys.notna()
    xs = xs[mask].astype(float)
    ys = ys[mask].astype(int)
    return xs, ys


def _direction_from_difference(x: pd.Series, y: pd.Series) -> int:
    if len(np.unique(y)) < 2:
        return 0
    pos = x[y == 1]
    neg = x[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0
    return 1 if float(pos.mean()) >= float(neg.mean()) else -1


def _safe_pearson(x, y) -> float:
    xs, ys = _valid_xy(x, y)
    if len(xs) < 3 or xs.nunique() < 2 or ys.nunique() < 2:
        return 0.0
    r = float(np.corrcoef(xs, ys)[0, 1])
    if not math.isfinite(r):
        return 0.0
    return r


def _safe_spearman(x, y) -> float:
    xs, ys = _valid_xy(x, y)
    if len(xs) < 3 or xs.nunique() < 2 or ys.nunique() < 2:
        return 0.0
    r, _ = spearmanr(xs, ys)
    r = float(r)
    if not math.isfinite(r):
        return 0.0
    return r


def _safe_auc(x, y):
    xs, ys = _valid_xy(x, y)
    if len(xs) < 3 or xs.nunique() < 2 or ys.nunique() < 2:
        return 0.5, 0.5, 0
    try:
        auc = float(roc_auc_score(ys, xs))
    except Exception:
        return 0.5, 0.5, 0
    oriented_auc = max(auc, 1.0 - auc)
    direction = 1 if auc >= 0.5 else -1
    return auc, oriented_auc, direction


def _safe_mannwhitney(x, y):
    xs, ys = _valid_xy(x, y)
    if len(xs) < 3 or xs.nunique() < 2 or ys.nunique() < 2:
        return 0.0, 1.0, 0.5, 0

    pos = xs[ys == 1]
    neg = xs[ys == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0, 1.0, 0.5, 0

    try:
        u, p = mannwhitneyu(pos, neg, alternative="two-sided")
        auc = float(u / (len(pos) * len(neg)))
        oriented_auc = max(auc, 1.0 - auc)
        rank_biserial_abs = abs(2.0 * auc - 1.0)
        direction = 1 if auc >= 0.5 else -1
        return float(rank_biserial_abs), float(p), float(oriented_auc), int(direction)
    except Exception:
        return 0.0, 1.0, 0.5, 0


def _lasso_selection(X: pd.DataFrame, y) -> list[dict]:
    """
    True L1 logistic feature selection.

    Previous versions set l1_ratio without penalty='l1'. Here the selector is
    intentionally explicit: penalty='l1', solver='liblinear'.
    """
    if X.shape[1] == 0:
        return []

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = LogisticRegression(
        solver="liblinear",
        l1_ratio=1.0,
        C=1.0,
        max_iter=5000,
    )

    model.fit(Xs, y)

    selected = []
    for name, coef in zip(X.columns, model.coef_[0]):
        coef = float(coef)
        if abs(coef) > 1e-6:
            selected.append({
                "feature": str(name),
                "method": "lasso",
                "score": abs(coef),
                "coef": coef,
                "direction": 1 if coef >= 0 else -1,
            })

    selected.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    return selected


def _mutual_info_selection(X: pd.DataFrame, y) -> list[dict]:
    if mutual_info_classif is None or X.shape[1] == 0:
        return []
    X_numeric = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X_numeric = X_numeric.fillna(X_numeric.median(numeric_only=True)).fillna(0)
    try:
        scores = mutual_info_classif(X_numeric, y, random_state=42)
    except Exception:
        return []
    out = []
    for name, score in zip(X.columns, scores):
        score = float(score)
        if not math.isfinite(score):
            score = 0.0
        out.append({
            "feature": str(name),
            "method": "mutual_info",
            "score": score,
            "mutual_info": score,
            "direction": _direction_from_difference(X[name], y),
        })
    out.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    return out


def select_features(
    X: pd.DataFrame,
    y,
    method: str = "lasso",
    top_n: int | None = None,
) -> list[dict]:
    """
    Select/rank features with a configurable method.

    Methods:
      - lasso: true L1 logistic selection, multivariate.
      - pearson: point-biserial Pearson correlation with binary outcome.
      - spearman: monotonic rank correlation with binary outcome.
      - auc: univariate ROC-AUC ranking, orientation-invariant.
      - mannwhitney: rank-biserial/Mann-Whitney ranking.
      - mutual_info: non-linear univariate dependency estimate.

    The function returns JSON-friendly dictionaries sorted from best to worst.
    """
    method = normalize_selection_method(method)

    if method == "lasso":
        out = _lasso_selection(X, y)
    elif method == "mutual_info":
        out = _mutual_info_selection(X, y)
    else:
        out = []
        for feature in X.columns:
            if method == "pearson":
                r = _safe_pearson(X[feature], y)
                item = {
                    "feature": str(feature),
                    "method": method,
                    "score": abs(r),
                    "correlation": r,
                    "direction": 1 if r >= 0 else -1,
                }
            elif method == "spearman":
                r = _safe_spearman(X[feature], y)
                item = {
                    "feature": str(feature),
                    "method": method,
                    "score": abs(r),
                    "correlation": r,
                    "direction": 1 if r >= 0 else -1,
                }
            elif method == "auc":
                auc, oriented_auc, direction = _safe_auc(X[feature], y)
                item = {
                    "feature": str(feature),
                    "method": method,
                    "score": oriented_auc,
                    "auc": auc,
                    "oriented_auc": oriented_auc,
                    "direction": direction,
                }
            elif method == "mannwhitney":
                effect, p, oriented_auc, direction = _safe_mannwhitney(X[feature], y)
                item = {
                    "feature": str(feature),
                    "method": method,
                    "score": effect,
                    "rank_biserial_abs": effect,
                    "p_value": p,
                    "oriented_auc": oriented_auc,
                    "direction": direction,
                }
            else:  # defensive only; normalize_selection_method already validates
                raise ValueError(f"Unsupported method: {method}")
            out.append(item)

        def _sort_key(item):
            # Most methods: higher score is better. For Mann-Whitney use p-value
            # as a tie-breaker after the effect size.
            return (
                float(item.get("score") or 0.0),
                -float(item.get("p_value") or 1.0),
            )

        out.sort(key=_sort_key, reverse=True)

    if top_n is not None:
        out = out[: int(top_n)]

    return out


def selection_method_label(method: str | None) -> str:
    method = normalize_selection_method(method)
    labels = {
        "lasso": "LASSO / L1 logistic",
        "pearson": "Pearson / point-biserial",
        "spearman": "Spearman rank correlation",
        "auc": "Univariate ROC-AUC",
        "mannwhitney": "Mann-Whitney / rank-biserial",
        "mutual_info": "Mutual information",
    }
    return labels[method]
