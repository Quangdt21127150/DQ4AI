"""
Module 1 of HybridDQR: Quality-Aware Diagnosis (QAD).

Per the paper, QAD scores a dataset on 6 quality dimensions -> a profile q in [0,1]^6
(Definition 1 in the paper; Section 4.1). Rather than re-deriving the six quality
formulas from scratch, this module reuses the exact, already-implemented and tested
`Polluter.compute_quality_measure` methods from the DQ4AI codebase (`polluters/*.py`),
since DQ4AI's own definitions ARE the six dimensions HybridDQR adopts (paper Section 3.1
explicitly: "We adopt the six data quality dimensions defined in [DQ4AI]").

In this experimental-validation setting (mirroring DQ4AI's own methodology exactly), a
dataset is polluted one dimension at a time. So for any given (dataset, active pollution
dimension) pair, the other 5 dimensions are known-by-construction to be at quality 1.0 --
nothing touched them. QAD therefore never needs a from-scratch, ground-truth-free scoring
path for the untouched dimensions; it only needs to read off the *active* dimension's
already-computed measure from the Polluter instance that produced the pollution.
"""
from typing import Dict, Tuple, Union

DIMENSIONS = [
    "consistency",
    "completeness",
    "feature_accuracy",
    "target_accuracy",
    "uniqueness",
    "class_balance",
]

# Maps each DQ4AI Polluter class name to the HybridDQR quality dimension it degrades.
POLLUTER_TO_DIMENSION = {
    "ConsistentRepresentationPolluter": "consistency",
    "CompletenessPolluter": "completeness",
    "FeatureAccuracyPolluter": "feature_accuracy",
    "TargetAccuracyPolluter": "target_accuracy",
    "UniquenessPolluter": "uniqueness",
    "ClassBalancePolluter": "class_balance",
}


def scalarize_quality_measure(dimension: str, measure: Union[float, Tuple]) -> float:
    """
    Some Polluter.compute_quality_measure() implementations return a tuple rather than a
    single float (consistency returns (overall, pollutable-only); feature_accuracy returns
    (categorical_quality, numerical_quality), either of which may be None if the dataset
    has no columns of that type). Reduce any of these to the single scalar q_i in [0,1]
    HybridDQR's profile expects.
    """
    if dimension == "consistency":
        # (overall_quality, pollutable_only_quality) -- Definition 2 in the DQ4AI paper
        # aggregates over ALL columns, so we report the overall figure.
        overall, _pollutable_only = measure
        return float(overall)
    if dimension == "feature_accuracy":
        cat_quality, num_quality = measure
        values = [v for v in (cat_quality, num_quality) if v is not None]
        return float(sum(values) / len(values)) if values else 1.0
    return float(measure)


def compute_quality_profile(active_dimension: str, quality_measure: Union[float, Tuple]) -> Dict[str, float]:
    """
    Build the 6-dimension quality profile q for a dataset that has had exactly one
    dimension polluted (the standard DQ4AI/HybridDQR experimental setup).

    :param active_dimension: one of DIMENSIONS -- the dimension that was actually polluted
    :param quality_measure: the raw return value of that dimension's
        Polluter.compute_quality_measure(polluted_df, original_df)
    :return: dict of the 6 dimensions -> quality in [0,1], with every dimension other than
        `active_dimension` fixed at 1.0 (untouched by construction)
    """
    assert active_dimension in DIMENSIONS, f"Unknown dimension: {active_dimension}"
    profile = {dim: 1.0 for dim in DIMENSIONS}
    profile[active_dimension] = scalarize_quality_measure(active_dimension, quality_measure)
    return profile
