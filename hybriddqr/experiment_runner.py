"""
Section 8's experimental protocol, run at a reduced scale for tractability: two datasets per
task, ~5 pollution levels (one random seed) per dimension instead of the full 0-100%/5-seed
sweep, lightweight-only candidate models (see robust_model_selection.py). Reuses DQ4AI's own
`Polluter.configure()` sweep and `compute_quality_measure` directly (see
quality_diagnosis.py's module docstring for why).

Run from the repo root:

    python -m hybriddqr.experiment_runner --task classification
    python -m hybriddqr.experiment_runner --task regression
    python -m hybriddqr.experiment_runner --task clustering
    python -m hybriddqr.experiment_runner            # all three, sequentially
"""
import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from polluters import (
    ConsistentRepresentationPolluter,
    CompletenessPolluter,
    FeatureAccuracyPolluter,
    TargetAccuracyPolluter,
    UniquenessPolluter,
    ClassBalancePolluter,
)
from regression.utils import discretize_column

from hybriddqr.baselines import run_all_baselines
from hybriddqr.policy import run_hybriddqr_pipeline
from hybriddqr.robust_model_selection import MODEL_REGISTRY, run_model

DATA_DIR = Path("data/clean")
RESULTS_DIR = Path("hybriddqr/re-results")

TASK_DATASETS = {
    "classification": ["SouthGermanCredit.csv", "cmc.data"],
    "regression": ["vw_prepared.csv", "imdb_prepared.csv"],
    "clustering": ["bank_2967137.csv", "covtype_2967137.csv"],
}

# Datasets whose base metadata entry (in metadata.json) is keyed under a different name than
# the actual sampled file on disk (clustering's per-seed files reuse the base dataset's
# metadata entry, same pattern as clustering/run_experiments.py).
TASK_METADATA_KEY = {
    "classification": {"SouthGermanCredit.csv": "SouthGermanCredit.csv", "cmc.data": "cmc.data"},
    "regression": {"vw_prepared.csv": "vw_prepared.csv", "imdb_prepared.csv": "imdb_prepared.csv"},
    "clustering": {"bank_2967137.csv": "bank.csv", "covtype_2967137.csv": "covtype.csv"},
}

DIMENSION_POLLUTERS = {
    "consistency": ConsistentRepresentationPolluter,
    "completeness": CompletenessPolluter,
    "feature_accuracy": FeatureAccuracyPolluter,
    "target_accuracy": TargetAccuracyPolluter,
    "uniqueness": UniquenessPolluter,
    "class_balance": ClassBalancePolluter,
}

DISCRETIZED_DIMENSIONS = {"uniqueness", "class_balance"}
N_LEVELS = 5


def load_metadata() -> dict:
    with open("metadata.json", "r") as f:
        return json.load(f)


def load_dataset(ds_name: str, metadata_entry: dict) -> pd.DataFrame:
    """Read a dataset and keep only the columns metadata.json actually declares
    (categorical_cols + numerical_cols + target). Some prepared CSVs carry extra columns
    metadata.json does not describe (e.g. imdb_prepared.csv has a free-text title column
    that is neither categorical nor numerical) -- passing those through to a model or
    one-hot encoder crashes on the undeclared dtype, so they are dropped here once, up
    front, rather than defensively at every call site."""
    df = pd.read_csv(DATA_DIR / ds_name)
    declared = metadata_entry.get("categorical_cols", []) + metadata_entry.get("numerical_cols", []) \
        + [metadata_entry["target"]]
    declared = [c for c in declared if c in df.columns]
    return df[declared]


def pick_levels(polluters: list, first_seed, n_levels: int = N_LEVELS) -> list:
    """Filter to a single random seed (this validation intentionally averages over levels,
    not seeds -- see module docstring), then take ~n_levels evenly-spaced instances across
    that seed's sweep (configure() builds the list with random_seed as the outermost loop,
    so a same-seed slice is contiguous and already ordered by increasing pollution level)."""
    same_seed = [p for p in polluters if p.random_seed == first_seed]
    if len(same_seed) <= n_levels:
        return same_seed
    indices = sorted(set(np.linspace(0, len(same_seed) - 1, n_levels).astype(int)))
    return [same_seed[i] for i in indices]


def prepare_for_pollution(df: pd.DataFrame, metadata_entry: dict, ds_name: str, task: str, dimension: str):
    """Regression's Uniqueness/ClassBalance polluters need a discretized (categorical-like)
    target column to group by -- same requirement Root code's own
    regression/run_regression_experiments.py works around. Returns (df_to_pollute,
    metadata_for_pollution, discretized_column_or_None)."""
    if task != "regression" or dimension not in DISCRETIZED_DIMENSIONS:
        return df, metadata_entry, None

    target = metadata_entry["target"]
    discr_col = "discr_" + target
    step = metadata_entry["discr_step_size"]
    bins = np.arange(df[target].min(), df[target].max() + step, step)
    df_discr = discretize_column(df, target, discr_col, list(bins))

    if dimension == "class_balance" and "class_balance_polluter_classes" in metadata_entry:
        # Mirror Root code's own run_regression_experiments.py: ClassBalancePolluter chokes
        # on discretized bins with very few samples (an "IndexError: list index out of
        # range" surfaces from its internal class-ordering math), so only the pre-selected,
        # well-populated bins are kept.
        keep_classes = metadata_entry["class_balance_polluter_classes"]
        df_discr = df_discr[df_discr[discr_col].isin(keep_classes)].reset_index(drop=True)

    discr_metadata = dict(metadata_entry)
    discr_metadata["target"] = discr_col
    return df_discr, discr_metadata, discr_col


def clean_reference_performance(task: str, df: pd.DataFrame, metadata_entry: dict) -> float:
    """Best clean-data performance across every candidate model for this task -- used to
    derive theta (target_performance) for the hybrid decision policy."""
    scores = []
    for model_key in MODEL_REGISTRY[task]:
        try:
            scores.append(run_model(task, model_key, df, metadata_entry))
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"clean_reference_performance: {model_key} failed: {exc}")
    return max(scores) if scores else 0.0


def to_native(obj):
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def run_dimension(task: str, dimension: str, df: pd.DataFrame, metadata_entry: dict, ds_name: str,
                   theta_fraction: float = 0.9) -> list:
    polluter_cls = DIMENSION_POLLUTERS[dimension]
    poll_df, poll_metadata, discr_col = prepare_for_pollution(df, metadata_entry, ds_name, task, dimension)

    metadata_for_configure = {"random_seeds": load_metadata()["random_seeds"], ds_name: poll_metadata}
    all_polluters = polluter_cls.configure(metadata_for_configure, poll_df, ds_name)
    first_seed = metadata_for_configure["random_seeds"][0]
    selected = pick_levels(all_polluters, first_seed)

    clean_perf = clean_reference_performance(task, df, metadata_entry)
    target_performance = theta_fraction * clean_perf

    results = []
    for polluter in selected:
        try:
            polluted_df = polluter.pollute(poll_df)
            raw_measure = polluter.compute_quality_measure(polluted_df, poll_df)
        except Exception as exc:
            warnings.warn(f"[{task}/{dimension}] pollute/measure failed, skipping level: {exc}")
            continue

        if discr_col is not None:
            polluted_df = polluted_df.drop(columns=[discr_col])

        def quality_measure_fn(candidate_df, _polluter=polluter, _poll_df=poll_df, _discr_col=discr_col):
            # Re-diagnose quality after repair. If a repair changed row count (dedup/SMOTE),
            # re-add the discretized column so the same grouping-based formulas still apply.
            df_for_measure = candidate_df
            if _discr_col is not None and _discr_col not in df_for_measure.columns:
                target = metadata_entry["target"]
                step = metadata_entry["discr_step_size"]
                bins = np.arange(df_for_measure[target].min(), df_for_measure[target].max() + step, step)
                df_for_measure = discretize_column(df_for_measure, target, _discr_col, list(bins))
            return _polluter.compute_quality_measure(df_for_measure, _poll_df)

        try:
            metric, quality_report = run_hybriddqr_pipeline(
                polluted_df, metadata_entry, task, dimension, raw_measure, quality_measure_fn, target_performance
            )
            baseline_metrics = run_all_baselines(polluted_df, metadata_entry, task, dimension)
        except Exception as exc:
            # A single pollution level misbehaving (e.g. a repair leaving a degenerate
            # class distribution at extreme pollution) shouldn't abort the whole dimension's
            # sweep -- log it and move on to the next level.
            warnings.warn(f"[{task}/{dimension}] level failed, skipping: {exc}")
            continue

        results.append(to_native({
            "pollution_params": polluter.get_pollution_params(),
            "quality_before": quality_report["quality_before"][dimension],
            "quality_after": quality_report["quality_after"][dimension],
            "decision": quality_report["decision"][dimension],
            "delta_estimate": quality_report["delta_estimates"][dimension],
            "model_selected": quality_report["model_selected"],
            "hybriddqr_metric": metric,
            **baseline_metrics,
        }))
        logging.info(f"[{task}/{dimension}] q={results[-1]['quality_before']:.3f} "
                     f"decision={results[-1]['decision']} hybriddqr={metric:.4f} "
                     f"B1={baseline_metrics['B1_no_repair_default']:.4f} "
                     f"B4={baseline_metrics['B4_oracle']:.4f}")
    return results


def run_task(task: str):
    metadata = load_metadata()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"hybriddqr_results_{task}.json"

    all_results = {}
    for ds_name in TASK_DATASETS[task]:
        metadata_key = TASK_METADATA_KEY[task][ds_name]
        metadata_entry = metadata[metadata_key]
        df = load_dataset(ds_name, metadata_entry)

        all_results[ds_name] = {}
        for dimension in DIMENSION_POLLUTERS:
            logging.info(f"=== {task}/{ds_name}: {dimension} ===")
            all_results[ds_name][dimension] = run_dimension(task, dimension, df, metadata_entry, metadata_key)

            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2, sort_keys=True)

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["classification", "regression", "clustering"], default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-5.5s] %(message)s",
                         stream=sys.stdout)

    tasks = [args.task] if args.task else ["classification", "regression", "clustering"]
    for task in tasks:
        run_task(task)


if __name__ == "__main__":
    main()
