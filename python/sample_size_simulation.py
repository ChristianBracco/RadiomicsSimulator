import numpy as np

from scipy.stats import norm

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    from analysis.sample_size_guardrail import analytical_guardrail_for_n
except ImportError:
    # Allows direct execution when this file is temporarily run outside package layout.
    from sample_size_guardrail import analytical_guardrail_for_n


# ============================================================
# EFFECT SIZE HELPERS
# ============================================================

def auc_to_cohens_d(auc):
    """
    Convert AUC into Cohen's d.

    This is used only for theoretical sample-size simulations.
    It does not use the real radiomic feature matrix.
    """

    auc = float(auc)

    if auc <= 0.5 or auc >= 1.0:
        raise ValueError(
            "auc_target must be > 0.5 and < 1.0."
        )

    return np.sqrt(2) * norm.ppf(
        auc
    )


# ============================================================
# SYNTHETIC DATASET SIMULATION
# ============================================================

def simulate_dataset(
    n_patients,
    auc_target,
    prevalence=0.45,
):
    """
    Create a synthetic one-feature binary dataset with an expected
    separation corresponding approximately to auc_target.

    Compared with the previous implementation, this version does not force
    50/50 classes. The number of events follows the requested prevalence.
    """

    n_patients = int(n_patients)
    prevalence = float(prevalence)

    if n_patients < 4:
        raise ValueError(
            "n_patients must be at least 4."
        )

    if not 0.0 < prevalence < 1.0:
        raise ValueError(
            "prevalence must be between 0 and 1."
        )

    sigma = 1.0

    d = auc_to_cohens_d(
        auc_target
    )

    n_events = int(
        round(n_patients * prevalence)
    )

    # Keep both classes represented, even for small N or extreme prevalence.
    n_events = max(
        1,
        min(n_patients - 1, n_events)
    )

    n_nonevents = (
        n_patients - n_events
    )

    x_events = np.random.normal(
        loc=d,
        scale=sigma,
        size=n_events
    )

    x_nonevents = np.random.normal(
        loc=0,
        scale=sigma,
        size=n_nonevents
    )

    X = np.concatenate(
        [
            x_events,
            x_nonevents
        ]
    ).reshape(
        -1,
        1
    )

    y = np.concatenate(
        [
            np.ones(n_events),
            np.zeros(n_nonevents)
        ]
    )

    # Randomize order so folds are not class-blocked.
    order = np.random.permutation(
        n_patients
    )

    return X[order], y[order]


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_dataset(
    X,
    y,
    n_splits=5,
    evaluation_mode="marker_auc",
):
    """
    Evaluate the synthetic dataset.

    Default mode is intentionally fast: for a one-dimensional synthetic
    biomarker, the raw marker AUC is equivalent to the ranking learned by a
    monotonic logistic model and is sufficient for Monte Carlo sample-size
    exploration.

    Use evaluation_mode="cv_logistic" only as a slow sensitivity check.
    """

    y = np.asarray(
        y
    )

    if len(np.unique(y)) < 2:
        return None

    if evaluation_mode == "marker_auc":
        return roc_auc_score(
            y,
            np.asarray(X).reshape(len(y), -1)[:, 0]
        )

    class_counts = np.bincount(
        y.astype(int),
        minlength=2
    )

    usable_splits = int(
        min(n_splits, class_counts.min())
    )

    if usable_splits < 2:
        return None

    cv = StratifiedKFold(
        n_splits=usable_splits,
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

    if len(np.unique(truths)) < 2:
        return None

    return roc_auc_score(
        truths,
        predictions
    )


# ============================================================
# MONTE CARLO + ANALYTICAL GUARDRAILS
# ============================================================

def sample_size_simulation(
    auc_target,
    n_simulations=1000,
    scenario_label=None,
    prevalence=0.45,
    risk_margin=0.10,
    r2_cs_adj=0.08,
    final_predictors_k=1,
    target_shrinkage=0.90,
    delta_nagelkerke=0.05,
    sample_sizes=None,
    evaluation_mode="marker_auc",
):
    """
    Theoretical Monte Carlo sample-size simulation.

    Important:
    - this function does not use the real radiomic feature matrix;
    - it simulates synthetic one-dimensional biomarkers;
    - observed model AUC can be passed as auc_target to create
      an observed-AUC scenario;
    - prevalence and event-rate precision are now explicit design inputs;
    - each simulated N is annotated with an analytical sample-size guardrail;
    - evaluation_mode="marker_auc" keeps the dashboard run fast.
    """

    prevalence = float(prevalence)

    if sample_sizes is None:
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
            200,
            250,
            300,
            350,
            400,
            500,
            600
        ]

    results = []

    if scenario_label:
        print(
            f"\nSIMULATION {scenario_label} | "
            f"AUC={auc_target:.3f} | prevalence={prevalence:.3f}"
        )
    else:
        print(
            f"\nSIMULATION AUC={auc_target:.2f} | prevalence={prevalence:.3f}"
        )

    for n in sample_sizes:

        aucs = []
        failed_runs = 0

        for _ in range(
            n_simulations
        ):
            X, y = simulate_dataset(
                n,
                auc_target,
                prevalence=prevalence
            )

            auc = evaluate_dataset(
                X,
                y,
                evaluation_mode=evaluation_mode
            )

            if auc is None:
                failed_runs += 1
                continue

            aucs.append(
                auc
            )

        if len(aucs) == 0:
            mean_auc = None
            std_auc = None
            ci95_low = None
            ci95_high = None
            power = None
        else:
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
                float(auc_target) - 0.05
            )

            power = float(
                np.mean(
                    aucs >= target_auc
                )
            )

        guardrail = analytical_guardrail_for_n(
            n_patients=int(n),
            prevalence=prevalence,
            risk_margin=risk_margin,
            r2_cs_adj=r2_cs_adj,
            final_predictors_k=final_predictors_k,
            target_shrinkage=target_shrinkage,
            delta_nagelkerke=delta_nagelkerke,
        )

        row = {
            "N": int(n),
            "mean_auc": mean_auc,
            "std_auc": std_auc,
            "ci95_low": ci95_low,
            "ci95_high": ci95_high,
            "power": power,
            "auc_target": float(auc_target),
            "success_threshold_auc": float(max(0.70, float(auc_target) - 0.05)),
            "scenario_label": scenario_label,
            "n_simulations_requested": int(n_simulations),
            "n_simulations_valid": int(len(aucs)) if isinstance(aucs, list) else int(aucs.size),
            "failed_runs": int(failed_runs),
            "evaluation_mode": str(evaluation_mode),
            "prevalence": float(prevalence),
            "risk_margin": float(risk_margin),
            "r2_cs_adj": float(r2_cs_adj),
            "final_predictors_k": int(final_predictors_k),
            "target_shrinkage": float(target_shrinkage),
            "delta_nagelkerke": float(delta_nagelkerke),
            "expected_events": guardrail["expected_events"],
            "expected_nonevents": guardrail["expected_nonevents"],
            "n_required_event_rate": guardrail["n_required_event_rate"],
            "n_required_shrinkage": guardrail["n_required_shrinkage"],
            "n_required_conservative": guardrail["n_required_conservative"],
            "k_max_raw": guardrail["k_max_raw"],
            "k_max_floor": guardrail["k_max_floor"],
            "passes_event_rate_guardrail": guardrail["passes_event_rate_guardrail"],
            "passes_predictor_guardrail": guardrail["passes_predictor_guardrail"],
            "passes_all_guardrails": guardrail["passes_all_guardrails"],
            "design_label": guardrail["design_label"],
        }

        results.append(
            row
        )

        auc_text = "NA" if mean_auc is None else f"{mean_auc:.3f}"
        power_text = "NA" if power is None else f"{power:.3f}"

        print(
            f"N={n:<3} | "
            f"events={row['expected_events']:<3} | "
            f"AUC={auc_text} | "
            f"POWER={power_text} | "
            f"k_max={row['k_max_raw']:.2f} | "
            f"design={row['design_label']}"
        )

    return results
