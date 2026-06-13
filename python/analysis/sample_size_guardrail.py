"""
Analytical sample-size guardrails for binary radiomics prediction models.

This module implements the practical checks inspired by the Riley/Cao
sample-size framework:

1) minimum N for precise estimation of the event rate / baseline risk;
2) minimum N for a target global shrinkage factor;
3) maximum number of predictors supported by a fixed N.

The Monte Carlo simulator remains empirical. These functions are intended
as methodological guardrails attached to each simulated N.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional


def _validate_probability(value: float, name: str) -> float:
    value = float(value)

    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}.")

    return value


def calculate_event_rate_sample_size(
    prevalence: float,
    risk_margin: float = 0.10,
) -> int:
    """
    Criterion III: minimum sample size for estimating the overall outcome
    risk / event rate with a given absolute margin of error.

    n = (1.96 / delta)^2 * phi * (1 - phi)
    """

    phi = _validate_probability(prevalence, "prevalence")
    delta = _validate_probability(risk_margin, "risk_margin")

    return int(math.ceil((1.96 / delta) ** 2 * phi * (1.0 - phi)))


def calculate_max_cox_snell_r2(prevalence: float) -> float:
    """
    The theoretical maximum Cox-Snell R2 for a binary endpoint with event
    prevalence phi.
    """

    phi = _validate_probability(prevalence, "prevalence")

    ln_lnull_per_patient = (
        phi * math.log(phi)
        + (1.0 - phi) * math.log(1.0 - phi)
    )

    return float(1.0 - math.exp(2.0 * ln_lnull_per_patient))


def calculate_required_shrinkage_for_optimism(
    prevalence: float,
    r2_cs_adj: float,
    delta_nagelkerke: float = 0.05,
) -> float:
    """
    Criterion II: minimum shrinkage needed to keep the absolute optimism
    in Nagelkerke R2 under delta_nagelkerke.
    """

    r2 = _validate_probability(r2_cs_adj, "r2_cs_adj")
    delta = _validate_probability(delta_nagelkerke, "delta_nagelkerke")
    max_r2_cs = calculate_max_cox_snell_r2(prevalence)

    return float(r2 / (r2 + delta * max_r2_cs))


def calculate_effective_shrinkage(
    prevalence: float,
    r2_cs_adj: float,
    target_shrinkage: float = 0.90,
    delta_nagelkerke: float = 0.05,
) -> float:
    """
    Use the stricter value between the target shrinkage and the shrinkage
    implied by Criterion II.
    """

    target = _validate_probability(target_shrinkage, "target_shrinkage")
    s_criterion_ii = calculate_required_shrinkage_for_optimism(
        prevalence=prevalence,
        r2_cs_adj=r2_cs_adj,
        delta_nagelkerke=delta_nagelkerke,
    )

    return float(max(target, s_criterion_ii))


def _predictor_denominator(
    prevalence: float,
    r2_cs_adj: float,
    target_shrinkage: float = 0.90,
    delta_nagelkerke: float = 0.05,
) -> float:
    """
    Positive denominator used by both:
    - n required for k predictors;
    - k allowed for n patients.
    """

    r2 = _validate_probability(r2_cs_adj, "r2_cs_adj")
    s_effective = calculate_effective_shrinkage(
        prevalence=prevalence,
        r2_cs_adj=r2,
        target_shrinkage=target_shrinkage,
        delta_nagelkerke=delta_nagelkerke,
    )

    if r2 >= s_effective:
        raise ValueError(
            "r2_cs_adj must be lower than the effective shrinkage factor. "
            f"Got r2_cs_adj={r2}, effective_shrinkage={s_effective}."
        )

    denominator = (s_effective - 1.0) * math.log(1.0 - r2 / s_effective)

    if denominator <= 0.0:
        raise ValueError(
            "Invalid shrinkage denominator. Check r2_cs_adj and shrinkage settings."
        )

    return float(denominator)


def calculate_shrinkage_sample_size(
    final_predictors_k: int,
    prevalence: float,
    r2_cs_adj: float = 0.08,
    target_shrinkage: float = 0.90,
    delta_nagelkerke: float = 0.05,
) -> int:
    """
    Criteria I + II: minimum sample size for k final predictors.
    """

    k = int(final_predictors_k)

    if k < 1:
        raise ValueError("final_predictors_k must be >= 1.")

    denominator = _predictor_denominator(
        prevalence=prevalence,
        r2_cs_adj=r2_cs_adj,
        target_shrinkage=target_shrinkage,
        delta_nagelkerke=delta_nagelkerke,
    )

    return int(math.ceil(k / denominator))


def calculate_max_predictors(
    n_patients: int,
    prevalence: float,
    r2_cs_adj: float = 0.08,
    target_shrinkage: float = 0.90,
    delta_nagelkerke: float = 0.05,
) -> Dict[str, float]:
    """
    Inverse calculation: maximum number of final predictors supported by
    a fixed sample size.
    """

    n = int(n_patients)

    if n < 1:
        raise ValueError("n_patients must be >= 1.")

    denominator = _predictor_denominator(
        prevalence=prevalence,
        r2_cs_adj=r2_cs_adj,
        target_shrinkage=target_shrinkage,
        delta_nagelkerke=delta_nagelkerke,
    )

    k_raw = n * denominator

    return {
        "k_max_raw": float(k_raw),
        "k_max_floor": int(max(0, math.floor(k_raw))),
    }


def analytical_guardrail_for_n(
    n_patients: int,
    prevalence: float,
    risk_margin: float = 0.10,
    r2_cs_adj: float = 0.08,
    final_predictors_k: int = 1,
    target_shrinkage: float = 0.90,
    delta_nagelkerke: float = 0.05,
) -> Dict[str, object]:
    """
    Combine event-rate and shrinkage/predictor criteria for a single N.
    """

    n = int(n_patients)
    phi = _validate_probability(prevalence, "prevalence")

    n_risk = calculate_event_rate_sample_size(
        prevalence=phi,
        risk_margin=risk_margin,
    )

    n_shrinkage = calculate_shrinkage_sample_size(
        final_predictors_k=final_predictors_k,
        prevalence=phi,
        r2_cs_adj=r2_cs_adj,
        target_shrinkage=target_shrinkage,
        delta_nagelkerke=delta_nagelkerke,
    )

    k_info = calculate_max_predictors(
        n_patients=n,
        prevalence=phi,
        r2_cs_adj=r2_cs_adj,
        target_shrinkage=target_shrinkage,
        delta_nagelkerke=delta_nagelkerke,
    )

    events = int(round(n * phi))
    nonevents = int(n - events)

    passes_event_rate = n >= n_risk
    passes_predictor = n >= n_shrinkage
    passes_all = passes_event_rate and passes_predictor

    if passes_all:
        label = "analytically_supported"
    elif passes_predictor:
        label = "predictor_ok_event_rate_uncertain"
    else:
        label = "exploratory_only"

    return {
        "n": int(n),
        "prevalence": float(phi),
        "expected_events": int(events),
        "expected_nonevents": int(nonevents),
        "risk_margin": float(risk_margin),
        "r2_cs_adj": float(r2_cs_adj),
        "target_shrinkage": float(target_shrinkage),
        "delta_nagelkerke": float(delta_nagelkerke),
        "final_predictors_k": int(final_predictors_k),
        "max_r2_cs": calculate_max_cox_snell_r2(phi),
        "s_min_criterion_ii": calculate_required_shrinkage_for_optimism(
            prevalence=phi,
            r2_cs_adj=r2_cs_adj,
            delta_nagelkerke=delta_nagelkerke,
        ),
        "s_effective": calculate_effective_shrinkage(
            prevalence=phi,
            r2_cs_adj=r2_cs_adj,
            target_shrinkage=target_shrinkage,
            delta_nagelkerke=delta_nagelkerke,
        ),
        "n_required_event_rate": int(n_risk),
        "n_required_shrinkage": int(n_shrinkage),
        "n_required_conservative": int(max(n_risk, n_shrinkage)),
        "k_max_raw": k_info["k_max_raw"],
        "k_max_floor": k_info["k_max_floor"],
        "passes_event_rate_guardrail": bool(passes_event_rate),
        "passes_predictor_guardrail": bool(passes_predictor),
        "passes_all_guardrails": bool(passes_all),
        "design_label": label,
    }


def build_sample_size_design_summary(
    current_n: int,
    prevalence: float,
    risk_margins: Optional[Iterable[float]] = None,
    r2_cs_adj_scenarios: Optional[Iterable[float]] = None,
    predictor_counts: Optional[Iterable[int]] = None,
    target_shrinkage: float = 0.90,
    delta_nagelkerke: float = 0.05,
) -> Dict[str, object]:
    """
    Compact summary to store in analysis_results.json and summary.txt.
    """

    if risk_margins is None:
        risk_margins = [0.05, 0.075, 0.10]

    if r2_cs_adj_scenarios is None:
        r2_cs_adj_scenarios = [0.05, 0.08, 0.113]

    if predictor_counts is None:
        predictor_counts = [1, 2, 3, 5, 10]

    phi = _validate_probability(prevalence, "prevalence")

    event_rate = []
    for margin in risk_margins:
        event_rate.append({
            "risk_margin": float(margin),
            "n_required": calculate_event_rate_sample_size(
                prevalence=phi,
                risk_margin=float(margin),
            ),
        })

    predictor_guardrails: List[Dict[str, object]] = []
    for r2 in r2_cs_adj_scenarios:
        r2_block = {
            "r2_cs_adj": float(r2),
            "k_max_current_n": calculate_max_predictors(
                n_patients=current_n,
                prevalence=phi,
                r2_cs_adj=float(r2),
                target_shrinkage=target_shrinkage,
                delta_nagelkerke=delta_nagelkerke,
            ),
            "n_required_by_predictor_count": [],
        }

        for k in predictor_counts:
            r2_block["n_required_by_predictor_count"].append({
                "final_predictors_k": int(k),
                "n_required_shrinkage": calculate_shrinkage_sample_size(
                    final_predictors_k=int(k),
                    prevalence=phi,
                    r2_cs_adj=float(r2),
                    target_shrinkage=target_shrinkage,
                    delta_nagelkerke=delta_nagelkerke,
                ),
            })

        predictor_guardrails.append(r2_block)

    return {
        "current_n": int(current_n),
        "prevalence": float(phi),
        "current_expected_events": int(round(current_n * phi)),
        "current_expected_nonevents": int(current_n - round(current_n * phi)),
        "target_shrinkage": float(target_shrinkage),
        "delta_nagelkerke": float(delta_nagelkerke),
        "event_rate_guardrails": event_rate,
        "predictor_guardrails": predictor_guardrails,
        "interpretation": (
            "Use Monte Carlo as the empirical performance estimate and these "
            "guardrails as methodological constraints on event-rate precision, "
            "overfitting and the maximum number of final predictors."
        ),
    }
