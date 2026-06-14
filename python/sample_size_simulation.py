import numpy as np

from scipy.stats import norm

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def auc_to_cohens_d(auc):
    """
    Convert AUC into Cohen's d.

    This is used only for theoretical sample-size simulations.
    It does not use the real radiomic feature matrix.
    """

    return np.sqrt(2) * norm.ppf(
        auc
    )


def simulate_dataset(
    n_patients,
    auc_target,
):
    """
    Create a synthetic one-feature binary dataset with an expected
    separation corresponding approximately to auc_target.
    """

    sigma = 1.0

    d = auc_to_cohens_d(
        auc_target
    )

    responders = (
        n_patients // 2
    )

    non_responders = (
        n_patients - responders
    )

    x_r = np.random.normal(
        loc=d,
        scale=sigma,
        size=responders
    )

    x_nr = np.random.normal(
        loc=0,
        scale=sigma,
        size=non_responders
    )

    X = np.concatenate(
        [
            x_r,
            x_nr
        ]
    ).reshape(
        -1,
        1
    )

    y = np.concatenate(
        [
            np.ones(responders),
            np.zeros(non_responders)
        ]
    )

    return X, y


def evaluate_dataset(
    X,
    y
):
    """
    Evaluate the synthetic dataset using a simple logistic regression.
    """

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    predictions = []
    truths = []

    for train_idx, test_idx in cv.split(
        X,
        y
    ):
        X_train = X[
            train_idx
        ]

        X_test = X[
            test_idx
        ]

        y_train = y[
            train_idx
        ]

        y_test = y[
            test_idx
        ]

        model = LogisticRegression(
            l1_ratio=0.0,
            max_iter=500
        )

        model.fit(
            X_train,
            y_train
        )

        prob = model.predict_proba(
            X_test
        )[:, 1]

        predictions.extend(
            prob
        )

        truths.extend(
            y_test
        )

    return roc_auc_score(
        truths,
        predictions
    )


def sample_size_simulation(
    auc_target,
    n_simulations=1000,
    scenario_label=None,
):
    """
    Theoretical Monte Carlo sample-size simulation.

    Important:
    - this function does not use the real radiomic feature matrix;
    - it simulates synthetic one-dimensional biomarkers;
    - observed model AUC can be passed as auc_target to create
      an observed-AUC scenario.
    """

    sample_sizes = [
        20,
        30,
        40,
        50,
        75,
        100,
        125,
        150,
        175,
        200
    ]

    results = []

    if scenario_label:
        print(
            f"\nSIMULATION {scenario_label} | AUC={auc_target:.3f}"
        )
    else:
        print(
            f"\nSIMULATION AUC={auc_target:.2f}"
        )

    for n in sample_sizes:

        aucs = []

        for _ in range(
            n_simulations
        ):
            X, y = simulate_dataset(
                n,
                auc_target
            )

            auc = evaluate_dataset(
                X,
                y
            )

            aucs.append(
                auc
            )

        aucs = np.array(
            aucs
        )

        mean_auc = float(
            aucs.mean()
        )

        std_auc = float(
            aucs.std()
        )

        ci95_low = float(
            np.percentile(
                aucs,
                2.5
            )
        )

        ci95_high = float(
            np.percentile(
                aucs,
                97.5
            )
        )

        # Operational power definition:
        # a study is counted as successful if the observed simulated AUC
        # is at least 0.05 below the target AUC, with a floor of 0.70.
        target_auc = max(
            0.70,
            auc_target - 0.05
        )

        power = float(
            np.mean(
                aucs >= target_auc
            )
        )

        results.append({
            "N": int(n),
            "mean_auc": mean_auc,
            "std_auc": std_auc,
            "ci95_low": ci95_low,
            "ci95_high": ci95_high,
            "power": power,
            "auc_target": float(auc_target),
            "success_threshold_auc": float(target_auc),
            "scenario_label": scenario_label
        })

        print(
            f"N={n:<3} | "
            f"AUC={mean_auc:.3f} | "
            f"POWER={power:.3f}"
        )

    return results
