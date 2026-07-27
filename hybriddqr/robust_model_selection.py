"""
Module 3 of HybridDQR: Robust Model Selection (RMS).

Implements Definition 3 (robustness ranking) / Table 4 and Eq. 3 (weighted robustness
score, WRS) from the paper, plus the adapter shims needed to instantiate and run DQ4AI's
existing per-task experiment classes uniformly (classification/regression/clustering each
have a different constructor signature and result shape).

The candidate pool sticks to lightweight, sklearn-backed classes only. `TabNet*`,
`Pytorch*`, and `AutoencoderExperiment` are excluded: these rely on PyTorch training loops
(fixed-epoch schedules, DataLoader-based batching, third-party TabNet internals) that are
substantially heavier and less predictable to run than the sklearn estimators used
elsewhere in this module, which is a poor fit for a diagnosis-and-repair pipeline meant to
run quickly and repeatedly. Regression has no genuinely-sklearn "Deep NN" candidate in this
codebase (`MLPRegressionExperiment`'s name is shadowed by a custom `torch.nn.Module` class
defined later in `regression/experiments.py`, so instantiating it does not give a plain
sklearn MLP), so regression's robustness ranking uses 3 families (Ensemble/Linear/Tree)
rather than 4. Clustering's ranking is not taken from Table 4 at all (its families --
centroid/hierarchical/density/probabilistic -- don't map onto Ensemble/Linear/Tree/DeepNN);
it is instead adapted from DQ4AI's own empirical findings (Section 7.1 of that paper: "the
k-means/k-prototypes algorithms... showed the most robustness regarding the six data
quality dimensions").
"""
from sklearn.model_selection import train_test_split

from classification.experiments import (
    LogRegExperiment,
    DecisionTreeExperiment,
    GradientBoostingClassifierExperiment,
    MultilayerPerceptronExperiment,
)
from regression.experiments import (
    RidgeRegressionExperiment,
    DecisionTreeRegressionExperiment,
    RandomForestRegressionExperiment,
)
import clustering.experiments as _clustering_experiments
from kmodes.kprototypes import KPrototypes as _KPrototypes


class _SingleProcessKPrototypes(_KPrototypes):
    """KMeansExperiment's k-Prototypes branch (clustering/experiments.py) hardcodes
    n_jobs=-1, which spawns a joblib/loky worker pool. On this Windows machine a stalled
    worker ("A worker stopped while some jobs were given to the executor") has been
    observed to block the whole pool for hours rather than erroring out -- the same class
    of Windows multiprocessing issue repair.py already works around for cleanlab/
    cross_val_predict by pinning n_jobs=1. Patching the module-level name here (rather than
    editing clustering/experiments.py, which is Root code) forces every KPrototypes
    instantiation to run single-process."""

    def __init__(self, *args, **kwargs):
        kwargs["n_jobs"] = 1
        super().__init__(*args, **kwargs)


_clustering_experiments.KPrototypes = _SingleProcessKPrototypes

from sklearn.mixture import GaussianMixture as _GaussianMixture


class _FastGaussianMixture(_GaussianMixture):
    """GaussianMixtureExperiment (clustering/experiments.py) fits with the sklearn default
    covariance_type='full' and n_init=10. Full covariance costs O(d^3) per EM step per
    component; on covtype_2967137.csv (55 one-hot columns, vs. bank's 4) this made a single
    pollution level take multiple hours instead of minutes. Diagonal covariance is the
    standard fix for exactly this case (cost drops to O(d)) and one init instead of ten cuts
    the remaining constant factor -- same rationale as the KPrototypes patch above: keep
    Root code's file untouched, adjust the runtime behaviour from here."""

    def __init__(self, n_components=1, *, covariance_type="full", tol=1e-3, reg_covar=1e-6,
                 max_iter=100, n_init=1, init_params="kmeans", weights_init=None,
                 means_init=None, precisions_init=None, random_state=None, warm_start=False,
                 verbose=0, verbose_interval=10):
        # Signature must mirror sklearn.mixture.GaussianMixture's exactly (no *args/**kwargs)
        # -- BaseEstimator.get_params() inspects __init__ and raises on varargs, which broke
        # every call silently (caught by the try/except around each model in
        # clean_reference_performance, so this model was quietly dropped instead of erroring
        # loudly). covariance_type/n_init are still forced to the fast settings below
        # regardless of what's passed in.
        super().__init__(
            n_components=n_components, covariance_type="diag", tol=tol, reg_covar=reg_covar,
            max_iter=max_iter, n_init=1, init_params=init_params, weights_init=weights_init,
            means_init=means_init, precisions_init=precisions_init, random_state=random_state,
            warm_start=warm_start, verbose=verbose, verbose_interval=verbose_interval,
        )


_clustering_experiments.GaussianMixture = _FastGaussianMixture

from clustering.experiments import KMeansExperiment, GaussianMixtureExperiment

CLASSIFICATION_MODELS = {
    "ensemble": GradientBoostingClassifierExperiment,
    "linear": LogRegExperiment,
    "tree": DecisionTreeExperiment,
    "deep_nn": MultilayerPerceptronExperiment,
}

REGRESSION_MODELS = {
    "ensemble": RandomForestRegressionExperiment,
    "linear": RidgeRegressionExperiment,
    "tree": DecisionTreeRegressionExperiment,
}

CLUSTERING_MODELS = {
    "centroid": KMeansExperiment,
    "probabilistic": GaussianMixtureExperiment,
}

MODEL_REGISTRY = {
    "classification": CLASSIFICATION_MODELS,
    "regression": REGRESSION_MODELS,
    "clustering": CLUSTERING_MODELS,
}

# Table 4: 1 = most robust, 4 = least robust. Regression omits "deep_nn" (see module docstring).
ROBUSTNESS_RANKING = {
    "classification": {
        "consistency": {"ensemble": 1, "linear": 2, "tree": 3, "deep_nn": 4},
        "completeness": {"ensemble": 1, "linear": 3, "tree": 2, "deep_nn": 4},
        "feature_accuracy": {"ensemble": 1, "linear": 2, "tree": 3, "deep_nn": 4},
        "target_accuracy": {"ensemble": 2, "linear": 3, "tree": 1, "deep_nn": 4},
        "uniqueness": {"ensemble": 1, "linear": 1, "tree": 2, "deep_nn": 3},
        "class_balance": {"ensemble": 2, "linear": 3, "tree": 1, "deep_nn": 4},
    },
    "regression": {
        "consistency": {"ensemble": 1, "linear": 2, "tree": 3},
        "completeness": {"ensemble": 1, "linear": 3, "tree": 2},
        "feature_accuracy": {"ensemble": 1, "linear": 2, "tree": 3},
        "target_accuracy": {"ensemble": 2, "linear": 3, "tree": 1},
        "uniqueness": {"ensemble": 1, "linear": 1, "tree": 2},
        "class_balance": {"ensemble": 2, "linear": 3, "tree": 1},
    },
    # Adapted from DQ4AI's own empirical findings (Section 7.1), not Table 4: k-means/
    # k-prototypes (centroid-based) were reported as consistently the most robust family
    # across dimensions, while Gaussian mixture is more sensitive since it assumes the data
    # was drawn from a single Gaussian distribution. Target accuracy is a non-issue for
    # both (unsupervised algorithms never see labels during fitting).
    "clustering": {
        "consistency": {"centroid": 1, "probabilistic": 2},
        "completeness": {"centroid": 1, "probabilistic": 2},
        "feature_accuracy": {"centroid": 1, "probabilistic": 2},
        "target_accuracy": {"centroid": 1, "probabilistic": 1},
        "uniqueness": {"centroid": 1, "probabilistic": 1},
        "class_balance": {"centroid": 1, "probabilistic": 2},
    },
}


def compute_wrs(quality_profile: dict, severity_thresholds: dict, task: str) -> dict:
    """Eq. 3: WRS(m) = sum_i w_i * 1[q_i < tau_i] * rank_i(m)^-1, w_i = 1 - q_i.
    Returns {model_key: wrs_score} for every candidate model of `task`."""
    ranking = ROBUSTNESS_RANKING[task]
    scores = {model_key: 0.0 for model_key in MODEL_REGISTRY[task]}
    for dimension, quality in quality_profile.items():
        if quality >= severity_thresholds[dimension]:
            continue
        weight = 1.0 - quality
        for model_key in scores:
            rank = ranking[dimension][model_key]
            scores[model_key] += weight * (1.0 / rank)
    return scores


def select_best_model(quality_profile: dict, severity_thresholds: dict, task: str) -> str:
    """Model with the highest WRS is selected as m* (paper, just below Eq. 3). If no
    dimension is below its threshold, WRS is 0 for everyone -- fall back to the dimension-
    agnostic top choice (rank 1 in the most quality dimensions)."""
    wrs = compute_wrs(quality_profile, severity_thresholds, task)
    if max(wrs.values()) > 0:
        return max(wrs, key=wrs.get)
    return top_ranked_model(task)


def top_ranked_model(task: str, dimension: str = None) -> str:
    """The single best (rank-1, ties broken by first match) model for one dimension, or --
    if no dimension is given -- the model with the lowest average rank across all
    dimensions. Used by B3 (robust-only baseline) and by the policy's rho_top(i) estimate."""
    ranking = ROBUSTNESS_RANKING[task]
    if dimension is not None:
        dim_ranking = ranking[dimension]
        return min(dim_ranking, key=dim_ranking.get)
    avg_rank = {
        model_key: sum(ranking[d][model_key] for d in ranking) / len(ranking)
        for model_key in MODEL_REGISTRY[task]
    }
    return min(avg_rank, key=avg_rank.get)


def run_model(task: str, model_key: str, df: "pd.DataFrame", metadata_entry: dict, seed: int = 42) -> float:
    """Instantiate and run the given candidate model, returning the single scalar metric
    HybridDQR/DQ4AI both report for that task (macro-F1 / R^2 / AMI)."""
    model_cls = MODEL_REGISTRY[task][model_key]

    if task == "classification":
        target = metadata_entry["target"]
        value_counts = df[target].value_counts()
        can_stratify = len(df) >= 10 and (value_counts >= 2).all()
        try:
            train_df, test_df = train_test_split(
                df, test_size=0.2, random_state=seed, stratify=df[target] if can_stratify else None
            )
        except ValueError:
            # A repair (e.g. unconditional B2, or SMOTE/dedup interacting badly at extreme
            # pollution levels) can leave a degenerate class distribution even a plain
            # random split can't handle gracefully at very small n; fall back further.
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)
        train_df, test_df = train_df.reset_index(drop=True), test_df.reset_index(drop=True)
        result = model_cls(train_df, test_df, metadata_entry).run()
        scoring = next(iter(result.values()))["scoring"]
        return scoring["macro avg"]["f1-score"]

    if task == "regression":
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)
        train_df, test_df = train_df.reset_index(drop=True), test_df.reset_index(drop=True)
        result = model_cls(
            train_df, test_df, metadata_entry["target"], metadata_entry.get("categorical_cols", [])
        ).run()
        return result["r2_score"]

    if task == "clustering":
        result = model_cls(None, df, metadata_entry).run(verbose=False)
        return result["mutual information"]["adj_mut_info"]

    raise ValueError(f"Unknown task: {task}")
