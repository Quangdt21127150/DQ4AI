"""
The Hybrid Decision Policy (Definition 4) and Algorithm 1 (full HybridDQR execution).

SEVERITY_THRESHOLDS are Table 2's defaults; REPAIR_COSTS map Table 3's qualitative
Low/Medium/High repair cost to a numeric scale comparable to Delta_i (both expressed as a
fractional change in the reference model's performance metric).
"""
from typing import Callable, Dict, Tuple

from hybriddqr.quality_diagnosis import DIMENSIONS, compute_quality_profile
from hybriddqr.repair import apply_repair
from hybriddqr.robust_model_selection import run_model, select_best_model, top_ranked_model

SEVERITY_THRESHOLDS = {
    "consistency": 0.80,
    "completeness": 0.60,
    "feature_accuracy": 0.70,
    "target_accuracy": 0.80,
    "uniqueness": 0.30,
    "class_balance": 0.50,
}

REPAIR_COSTS = {
    "consistency": 0.1,       # Low  -- entity normalization
    "completeness": 0.3,      # Medium -- iterative imputation
    "feature_accuracy": 0.1,  # Low -- Z-score winsorization
    "target_accuracy": 0.6,   # High -- confident learning
    "uniqueness": 0.1,        # Low -- hash dedup
    "class_balance": 0.3,     # Medium -- SMOTE/undersampling
}

# Reference model used to estimate Delta_i (Eq. 2). Paper: "e.g., a linear classifier for
# classification tasks"; we extend the same "cheap, fast, task-appropriate" spirit to
# regression (Ridge) and clustering (KMeans, since there's no supervised reference there).
REFERENCE_MODEL_KEY = {"classification": "linear", "regression": "linear", "clustering": "centroid"}


def decide(
    df: "pd.DataFrame",
    metadata_entry: dict,
    task: str,
    quality_profile: Dict[str, float],
    target_performance: float,
) -> Tuple[Dict[str, str], Dict[str, float]]:
    """
    Definition 4: for every dimension, decide skip / repair / robust / both.

    :param target_performance: theta -- the minimum acceptable performance (already
        computed by the caller as, e.g., 0.9 * clean-data performance)
    :return: (decision_vector, delta_estimates)
    """
    reference_key = REFERENCE_MODEL_KEY[task]
    rho_before = run_model(task, reference_key, df, metadata_entry)

    decision, deltas = {}, {}
    for dimension in DIMENSIONS:
        quality = quality_profile[dimension]
        if quality >= SEVERITY_THRESHOLDS[dimension]:
            decision[dimension] = "skip"
            deltas[dimension] = 0.0
            continue

        try:
            repaired_df = apply_repair(dimension, df, metadata_entry, task)
            rho_after = run_model(task, reference_key, repaired_df, metadata_entry)
        except Exception:
            rho_after = rho_before
        delta = rho_after - rho_before
        deltas[dimension] = delta

        cost = REPAIR_COSTS[dimension]
        if delta <= cost:
            decision[dimension] = "robust"
            continue

        rho_top = run_model(task, top_ranked_model(task, dimension), df, metadata_entry)
        decision[dimension] = "both" if rho_top >= target_performance else "repair"

    return decision, deltas


def run_hybriddqr_pipeline(
    df_polluted: "pd.DataFrame",
    metadata_entry: dict,
    task: str,
    active_dimension: str,
    raw_quality_measure,
    quality_measure_fn: Callable[["pd.DataFrame"], float],
    target_performance: float,
) -> Tuple[float, dict]:
    """
    Algorithm 1, specialised to this validation's single-active-dimension experimental
    setting (matching DQ4AI's own one-dimension-at-a-time pollution methodology -- see
    quality_diagnosis.py's module docstring for why the other 5 dimensions are fixed at 1.0
    and therefore always decide "skip").

    :param raw_quality_measure: the raw (possibly tuple) return value of
        Polluter.compute_quality_measure(df_polluted, original_df) for `active_dimension`
    :param quality_measure_fn: callable(df) -> raw quality measure for `active_dimension`,
        bound to the same Polluter instance and the original clean dataframe, used to
        re-diagnose quality after repair (q_hat)
    :return: (final_metric, quality_report) where quality_report mirrors the paper's Q =
        (q, q_hat, d, m*)
    """
    from hybriddqr.quality_diagnosis import scalarize_quality_measure

    quality_before = compute_quality_profile(active_dimension, raw_quality_measure)
    decision, deltas = decide(df_polluted, metadata_entry, task, quality_before, target_performance)

    action = decision[active_dimension]
    if action in ("repair", "both"):
        repaired_df = apply_repair(active_dimension, df_polluted, metadata_entry, task)
        residual_measure = quality_measure_fn(repaired_df)
        quality_after = compute_quality_profile(active_dimension, residual_measure)
    else:
        repaired_df = df_polluted
        quality_after = quality_before

    model_selected = select_best_model(quality_after, SEVERITY_THRESHOLDS, task)
    metric = run_model(task, model_selected, repaired_df, metadata_entry)

    quality_report = {
        "quality_before": quality_before,
        "quality_after": quality_after,
        "decision": decision,
        "delta_estimates": deltas,
        "model_selected": model_selected,
    }
    return metric, quality_report
