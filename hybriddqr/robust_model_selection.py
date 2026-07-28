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

# --- Optional fixes for the previously-excluded TabNet / Pytorch* / AutoencoderExperiment
# families (see module docstring above for why they were excluded). These monkeypatches make
# each class safe to instantiate on this machine, but are deliberately NOT wired into
# MODEL_REGISTRY/ROBUSTNESS_RANKING below -- adding them there would change RMS's candidate
# pool and require a full rerun to take effect, which is out of scope for now. Kept here so a
# future run can opt in by importing e.g. _SafeMultilayerPerceptronPytorchExperiment instead
# of the sklearn-only classes above.

import torch.utils.data as _torch_data
import classification.experiments as _classification_experiments
import regression.experiments as _regression_experiments


class _SafeDataLoader(_torch_data.DataLoader):
    """Every Pytorch*Experiment class in classification/experiments.py and
    regression/experiments.py (KNN, MLP x3, SVM) hardcodes num_workers=36 in its DataLoader
    construction. On this Windows machine (12 cores) DataLoader workers are spawned via the
    'spawn' start method (no fork on Windows), which re-imports the whole module per worker;
    36 workers x 2 loaders (train+test) x pin_memory=True x prefetch_factor=2 reliably hung
    indefinitely in this session -- an oversubscription/resource-contention deadlock, not a
    crash. Forcing num_workers=0 (single-process loading) removes the deadlock; these
    datasets are at most ~15k rows so the throughput cost is negligible."""

    def __init__(self, *args, **kwargs):
        kwargs["num_workers"] = 0
        kwargs.pop("prefetch_factor", None)  # torch raises if this is set together with num_workers=0
        kwargs.pop("persistent_workers", None)
        super().__init__(*args, **kwargs)


_classification_experiments.DataLoader = _SafeDataLoader
_regression_experiments.DataLoader = _SafeDataLoader

# Pre-existing Root code bug, unrelated to the DataLoader deadlock above and not something
# the smoke test's other failures were expected to reveal: PytorchMLPClassifier.__init__
# requires `random_state` as its first positional argument, but PytorchMLPExperiment.__init__
# (classification/experiments.py) instantiates it with only keyword args (input_dim=...,
# hidden_layer_sizes=..., activation=..., output_dim=...) and never passes random_state --
# every call raises "missing 1 required positional argument: 'random_state'" before any
# DataLoader or training code runs. The class already calls set_random_seed(42) internally
# regardless of what's passed, so random_state is accepted but unused -- giving it a default
# makes the existing call site work without changing behavior.
#
# This must patch __init__ IN PLACE on the existing class object rather than subclassing and
# reassigning classification.experiments.PytorchMLPClassifier to the subclass (the same
# mistake the TabNet fix above avoids for a different reason). PytorchMLPClassifier.__init__
# itself calls `super(PytorchMLPClassifier, self).__init__()` -- a lookup of the *global name*
# PytorchMLPClassifier in this module's own namespace at call time, not a closure over the
# class object. Reassigning that module attribute to a subclass makes this internal call
# resolve super(<the subclass>, self), which walks the MRO to the ORIGINAL class and re-invokes
# ITS __init__ with zero arguments -- reproducing the exact "missing random_state and
# input_dim" crash one level down. Patching the method in place leaves the module's
# PytorchMLPClassifier name pointing at the original, unrenamed class, so that internal
# super() call still resolves to nn.Module as intended.
_original_pytorch_mlp_classifier_init = _classification_experiments.PytorchMLPClassifier.__init__


def _patched_pytorch_mlp_classifier_init(self, random_state=42, *args, **kwargs):
    return _original_pytorch_mlp_classifier_init(self, random_state, *args, **kwargs)


_classification_experiments.PytorchMLPClassifier.__init__ = _patched_pytorch_mlp_classifier_init

from pytorch_tabnet.tab_model import TabNetClassifier as _TabNetClassifier, TabNetRegressor as _TabNetRegressor


def _safe_tabnet_batch_kwargs(n_rows: int, kwargs: dict) -> dict:
    """pytorch_tabnet's fit() defaults to batch_size=1024, virtual_batch_size=128,
    drop_last=True. Two crash patterns are known independent of the num_workers deadlock
    above: (1) if n_rows < batch_size, drop_last=True discards the entire dataset, leaving
    zero batches; (2) virtual_batch_size must evenly divide batch_size (Ghost Batch Norm),
    else pytorch_tabnet raises directly. This shrinks batch_size to fit the actual dataset
    and collapses virtual_batch_size to match when it wouldn't divide evenly, avoiding both.
    NOTE: this session's earlier "TabNet internal RuntimeError" was not re-isolated to one
    exact cause before this fix was written -- treat this as a defensive guard against the
    two known small-batch failure modes above, not a verified reproduction of that error."""
    kwargs = dict(kwargs)
    batch_size = min(kwargs.get("batch_size", 1024), max(1, n_rows))
    virtual_batch_size = kwargs.get("virtual_batch_size", 128)
    if virtual_batch_size > batch_size or batch_size % virtual_batch_size != 0:
        virtual_batch_size = batch_size
    kwargs["batch_size"] = batch_size
    kwargs["virtual_batch_size"] = virtual_batch_size
    return kwargs


# Patch fit() in place on the ORIGINAL classes rather than subclassing + swapping the module
# attribute. Root code checks `self.model.__class__.__name__ == 'TabNetClassifier'` (and
# 'TabNetRegressor') to decide whether to convert X/y to .values before calling fit() --
# subclassing would rename the class and silently break that string check, sending a raw
# pandas DataFrame into pytorch_tabnet and crashing with "Pandas DataFrame are not supported"
# (caught by the smoke test below). Patching the bound method preserves the class identity.
_original_tabnet_classifier_fit = _TabNetClassifier.fit
_original_tabnet_regressor_fit = _TabNetRegressor.fit


def _safe_tabnet_classifier_fit(self, X_train, y_train, *args, **kwargs):
    kwargs = _safe_tabnet_batch_kwargs(len(X_train), kwargs)
    return _original_tabnet_classifier_fit(self, X_train, y_train, *args, **kwargs)


def _safe_tabnet_regressor_fit(self, X_train, y_train, *args, **kwargs):
    kwargs = _safe_tabnet_batch_kwargs(len(X_train), kwargs)
    return _original_tabnet_regressor_fit(self, X_train, y_train, *args, **kwargs)


_TabNetClassifier.fit = _safe_tabnet_classifier_fit
_TabNetRegressor.fit = _safe_tabnet_regressor_fit

import clustering.experiments as _clustering_experiments_mod
from clustering.experiments import AutoencoderExperiment as _AutoencoderExperiment


class _FastAutoencoderExperiment(_AutoencoderExperiment):
    """AutoencoderExperiment.run() (clustering/experiments.py) hardcodes 200 training epochs
    with no GPU available on this machine (torch.cuda.is_available() is False here), and its
    own DataLoaders already use num_workers=0 by default -- so unlike the Pytorch*Experiment
    classes above, this one was never at deadlock risk. Its problem is pure wall-clock cost:
    200 epochs x every pollution level x every seed is not tractable in the time available.
    Since the epoch count is a literal inside run() rather than a constructor parameter, the
    only way to change it without editing Root code is to override the whole method; the body
    below is otherwise identical to the original, just with the epoch count parameterized and
    lowered. 30 epochs is a reconstruction-loss autoencoder on a small dense net (2D embedding
    bottleneck) -- expected to have converged well before then, but this has not been verified
    against the original 200-epoch results, so treat any resulting numbers as approximate
    until spot-checked against a 200-epoch run."""

    N_EPOCHS = 30

    def run(self) -> dict:
        import logging as _logging
        from copy import deepcopy as _deepcopy

        import numpy as _np
        import pandas as _pd
        import torch as _torch
        from torch import nn as _nn, optim as _optim
        from torch.utils.data import DataLoader as _DataLoader, random_split as _random_split
        from tqdm import tqdm as _tqdm

        _logging.info(f"Running: {self.name} experiment (fast, {self.N_EPOCHS} epochs) ...")

        dataset = _clustering_experiments_mod.CustomDataset(self.test, self._target_col)
        trainsize = int(0.8 * len(dataset))
        trainset, testset = _random_split(dataset, [trainsize, len(dataset) - trainsize])

        trainloader = _DataLoader(trainset, batch_size=128, shuffle=True)
        testloader = _DataLoader(testset, batch_size=128, shuffle=False)

        optimizer = _optim.Adam(self.model.parameters(), lr=0.003)
        criterion = _nn.MSELoss()

        loss_per_batch = []
        running_loss = 0.0
        i = 0
        pbar = _tqdm(range(self.N_EPOCHS))

        self.model.train()
        for epoch in pbar:
            pbar.set_description(f"Epoch {epoch:03d}: Loss: {_np.sqrt(running_loss / (i + 1)):05f}")
            running_loss = 0.0
            for i, (inputs, _label) in enumerate(trainloader):
                inputs = inputs.to(self.device)
                optimizer.zero_grad()
                _, outputs = self.model(inputs.float())
                loss = criterion(outputs, inputs.float())
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            loss_per_batch.append(running_loss / (i + 1))

        self.model.eval()
        with _torch.no_grad():
            inputs, _labels = next(iter(testloader))
            inputs = inputs.to(self.device)
            _, outputs = self.model(inputs.float())
            _logging.info(
                f"RMSE Training: {_np.sqrt(loss_per_batch[-1])}, "
                f"Test: {_np.sqrt(criterion(outputs, inputs.float()).detach().cpu().numpy())}"
            )

        fullloader = _DataLoader(dataset, batch_size=128, shuffle=False)
        encoded_outputs = []
        with _torch.no_grad():
            for inputs, _label in fullloader:
                inputs = inputs.to(self.device)
                optimizer.zero_grad()
                encoded, _ = self.model(inputs.float())
                encoded_outputs.append(encoded)
        encoded_result = _torch.cat(encoded_outputs, dim=0).cpu().numpy()

        encoded_data = _pd.DataFrame(encoded_result)
        encoded_data[self._target_col] = self.test[self._target_col]

        adapted_metadata = _deepcopy(self.metadata)
        adapted_metadata["categorical_cols"] = []
        adapted_metadata["numerical_cols"] = encoded_data.columns.tolist()

        clustering_exp = GaussianMixtureExperiment(None, encoded_data, adapted_metadata)
        return clustering_exp.run(verbose=False)


_clustering_experiments_mod.AutoencoderExperiment = _FastAutoencoderExperiment

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
