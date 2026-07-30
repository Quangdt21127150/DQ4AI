"""
Module 3 of HybridDQR: Robust Model Selection (RMS).

Implements Definition 3 (robustness ranking) / Table 4 and Eq. 3 (weighted robustness
score, WRS) from the paper, plus the adapter shims needed to instantiate and run DQ4AI's
existing per-task experiment classes uniformly (classification/regression/clustering each
have a different constructor signature and result shape).

The candidate pool includes both the lightweight sklearn-backed classes and
`TabNet*`/`Pytorch*`/`AutoencoderExperiment` (`pytorch_mlp`/`tabnet`/`autoencoder` keys
below). Clustering's ranking is not taken from Table 4 at all (its families --
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
    TabNetExperiment,
    PytorchMLPExperiment,
)
from regression.experiments import (
    RidgeRegressionExperiment,
    DecisionTreeRegressionExperiment,
    RandomForestRegressionExperiment,
    TabNetRegressionExperiment,
    PytorchMLPRegressionExperiment,
)
import clustering.experiments as _clustering_experiments
from kmodes.kprototypes import KPrototypes as _KPrototypes


class _SingleProcessKPrototypes(_KPrototypes):
    def __init__(self, *args, **kwargs):
        kwargs["n_jobs"] = 1
        super().__init__(*args, **kwargs)


_clustering_experiments.KPrototypes = _SingleProcessKPrototypes

from sklearn.mixture import GaussianMixture as _GaussianMixture


class _FastGaussianMixture(_GaussianMixture):
    def __init__(
        self,
        n_components=1,
        *,
        covariance_type="full",
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        weights_init=None,
        means_init=None,
        precisions_init=None,
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10,
    ):
        super().__init__(
            n_components=n_components,
            covariance_type="diag",
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=1,
            init_params=init_params,
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init,
            random_state=random_state,
            warm_start=warm_start,
            verbose=verbose,
            verbose_interval=verbose_interval,
        )


_clustering_experiments.GaussianMixture = _FastGaussianMixture

from clustering.experiments import KMeansExperiment, GaussianMixtureExperiment

import torch.utils.data as _torch_data
import classification.experiments as _classification_experiments
import regression.experiments as _regression_experiments


class _SafeDataLoader(_torch_data.DataLoader):
    def __init__(self, *args, **kwargs):
        kwargs["num_workers"] = 0
        kwargs.pop("prefetch_factor", None)
        kwargs.pop("persistent_workers", None)
        super().__init__(*args, **kwargs)


_classification_experiments.DataLoader = _SafeDataLoader
_regression_experiments.DataLoader = _SafeDataLoader

_original_pytorch_mlp_classifier_init = (
    _classification_experiments.PytorchMLPClassifier.__init__
)


def _patched_pytorch_mlp_classifier_init(self, random_state=42, *args, **kwargs):
    return _original_pytorch_mlp_classifier_init(self, random_state, *args, **kwargs)


_classification_experiments.PytorchMLPClassifier.__init__ = (
    _patched_pytorch_mlp_classifier_init
)

from pytorch_tabnet.tab_model import (
    TabNetClassifier as _TabNetClassifier,
    TabNetRegressor as _TabNetRegressor,
)


def _safe_tabnet_batch_kwargs(n_rows: int, kwargs: dict) -> dict:
    kwargs = dict(kwargs)
    batch_size = min(kwargs.get("batch_size", 1024), max(1, n_rows))
    virtual_batch_size = kwargs.get("virtual_batch_size", 128)
    if virtual_batch_size > batch_size or batch_size % virtual_batch_size != 0:
        virtual_batch_size = batch_size
    kwargs["batch_size"] = batch_size
    kwargs["virtual_batch_size"] = virtual_batch_size
    return kwargs


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
    N_EPOCHS = 30

    def run(self, verbose=False) -> dict:
        import logging as _logging
        from copy import deepcopy as _deepcopy

        import numpy as _np
        import pandas as _pd
        import torch as _torch
        from torch import nn as _nn, optim as _optim
        from torch.utils.data import (
            DataLoader as _DataLoader,
            random_split as _random_split,
        )
        from tqdm import tqdm as _tqdm

        _logging.info(
            f"Running: {self.name} experiment (fast, {self.N_EPOCHS} epochs) ..."
        )

        dataset = _clustering_experiments_mod.CustomDataset(self.test, self._target_col)
        trainsize = int(0.8 * len(dataset))
        trainset, testset = _random_split(
            dataset, [trainsize, len(dataset) - trainsize]
        )

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
            pbar.set_description(
                f"Epoch {epoch:03d}: Loss: {_np.sqrt(running_loss / (i + 1)):05f}"
            )
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
    "pytorch_mlp": PytorchMLPExperiment,
    "tabnet": TabNetExperiment,
}

REGRESSION_MODELS = {
    "ensemble": RandomForestRegressionExperiment,
    "linear": RidgeRegressionExperiment,
    "tree": DecisionTreeRegressionExperiment,
    "deep_nn": PytorchMLPRegressionExperiment,
    "tabnet": TabNetRegressionExperiment,
}

CLUSTERING_MODELS = {
    "centroid": KMeansExperiment,
    "probabilistic": GaussianMixtureExperiment,
    "autoencoder": _FastAutoencoderExperiment,
}

MODEL_REGISTRY = {
    "classification": CLASSIFICATION_MODELS,
    "regression": REGRESSION_MODELS,
    "clustering": CLUSTERING_MODELS,
}

# Table 4: 1 = most robust, 4 = least robust.
ROBUSTNESS_RANKING = {
    "classification": {
        "consistency": {
            "ensemble": 1,
            "linear": 2,
            "tree": 3,
            "deep_nn": 4,
            "pytorch_mlp": 4,
            "tabnet": 4,
        },
        "completeness": {
            "ensemble": 1,
            "linear": 3,
            "tree": 2,
            "deep_nn": 4,
            "pytorch_mlp": 4,
            "tabnet": 4,
        },
        "feature_accuracy": {
            "ensemble": 1,
            "linear": 2,
            "tree": 3,
            "deep_nn": 4,
            "pytorch_mlp": 4,
            "tabnet": 4,
        },
        "target_accuracy": {
            "ensemble": 2,
            "linear": 3,
            "tree": 1,
            "deep_nn": 4,
            "pytorch_mlp": 4,
            "tabnet": 4,
        },
        "uniqueness": {
            "ensemble": 1,
            "linear": 1,
            "tree": 2,
            "deep_nn": 3,
            "pytorch_mlp": 3,
            "tabnet": 3,
        },
        "class_balance": {
            "ensemble": 2,
            "linear": 3,
            "tree": 1,
            "deep_nn": 4,
            "pytorch_mlp": 4,
            "tabnet": 4,
        },
    },
    "regression": {
        "consistency": {
            "ensemble": 1,
            "linear": 2,
            "tree": 3,
            "deep_nn": 4,
            "tabnet": 4,
        },
        "completeness": {
            "ensemble": 1,
            "linear": 3,
            "tree": 2,
            "deep_nn": 4,
            "tabnet": 4,
        },
        "feature_accuracy": {
            "ensemble": 1,
            "linear": 2,
            "tree": 3,
            "deep_nn": 4,
            "tabnet": 4,
        },
        "target_accuracy": {
            "ensemble": 2,
            "linear": 3,
            "tree": 1,
            "deep_nn": 4,
            "tabnet": 4,
        },
        "uniqueness": {
            "ensemble": 1,
            "linear": 1,
            "tree": 2,
            "deep_nn": 3,
            "tabnet": 3,
        },
        "class_balance": {
            "ensemble": 2,
            "linear": 3,
            "tree": 1,
            "deep_nn": 4,
            "tabnet": 4,
        },
    },
    "clustering": {
        "consistency": {"centroid": 1, "probabilistic": 2, "autoencoder": 3},
        "completeness": {"centroid": 1, "probabilistic": 2, "autoencoder": 3},
        "feature_accuracy": {"centroid": 1, "probabilistic": 2, "autoencoder": 3},
        "target_accuracy": {"centroid": 1, "probabilistic": 1, "autoencoder": 2},
        "uniqueness": {"centroid": 1, "probabilistic": 1, "autoencoder": 2},
        "class_balance": {"centroid": 1, "probabilistic": 2, "autoencoder": 3},
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


def select_best_model(
    quality_profile: dict, severity_thresholds: dict, task: str
) -> str:
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


def run_model(
    task: str, model_key: str, df: "pd.DataFrame", metadata_entry: dict, seed: int = 42
) -> float:
    """Instantiate and run the given candidate model, returning the single scalar metric
    HybridDQR/DQ4AI both report for that task (macro-F1 / R^2 / AMI)."""
    model_cls = MODEL_REGISTRY[task][model_key]

    if task == "classification":
        target = metadata_entry["target"]
        value_counts = df[target].value_counts()
        can_stratify = len(df) >= 10 and (value_counts >= 2).all()
        try:
            train_df, test_df = train_test_split(
                df,
                test_size=0.2,
                random_state=seed,
                stratify=df[target] if can_stratify else None,
            )
        except ValueError:
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)
        train_df, test_df = train_df.reset_index(drop=True), test_df.reset_index(
            drop=True
        )
        result = model_cls(train_df, test_df, metadata_entry).run()
        scoring = next(iter(result.values()))["scoring"]
        return scoring["macro avg"]["f1-score"]

    if task == "regression":
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)
        train_df, test_df = train_df.reset_index(drop=True), test_df.reset_index(
            drop=True
        )
        result = model_cls(
            train_df,
            test_df,
            metadata_entry["target"],
            metadata_entry.get("categorical_cols", []),
        ).run()
        if "r2_score" in result:
            return result["r2_score"]
        return next(iter(result.values()))["scoring"]["r2_score"]

    if task == "clustering":
        result = model_cls(None, df, metadata_entry).run(verbose=False)
        return result["mutual information"]["adj_mut_info"]

    raise ValueError(f"Unknown task: {task}")
