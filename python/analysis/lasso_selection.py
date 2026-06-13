from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def lasso_selection(
    X,
    y
):

    scaler = StandardScaler()

    Xs = scaler.fit_transform(X)

    model = LogisticRegression(
        l1_ratio=1.0,
        solver="liblinear",
        C=1.0,
        max_iter=5000
    )

    model.fit(
        Xs,
        y
    )

    selected = []

    for name, coef in zip(
        X.columns,
        model.coef_[0]
    ):

        if abs(coef) > 1e-6:

            selected.append({

                "feature": name,

                "coef": float(coef)
            })

    selected.sort(
        key=lambda x:
        abs(x["coef"]),
        reverse=True
    )

    return selected