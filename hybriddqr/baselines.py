"""
Section 8.3's four baselines, run against the same polluted dataset HybridDQR sees.

B1 (no repair, default model): the paper's example is "logistic regression for
classification"; we use each task's "linear" candidate uniformly (Ridge for regression,
KMeans for clustering) as the task-appropriate "default model".

B2 (full repair, default model): all 6 repair operators applied unconditionally -- including
to the 5 dimensions that (in this single-active-dimension experimental setting, see
quality_diagnosis.py) are already at quality 1.0. This is intentional, not wasted work: it's
exactly the dynamic the paper motivates HybridDQR against -- blanket repair can spend cost
(and risk introducing its own distortion, e.g. winsorizing already-clean numeric data) on
dimensions that didn't need it, which selective/cost-aware repair (HybridDQR itself) avoids.

B3 (no repair, best robust model): the single top-ranked model (Table-4-style ranking, see
robust_model_selection.py) for the dimension that's actually degraded, on unrepaired data.

B4 (oracle hybrid): brute-force over every repair variant this module can produce
(unrepaired, single-dimension repair of the one actually-degraded dimension, and B2's full
all-6-dimension repair) x every candidate model for the task, reporting the best
actually-measured metric. This must include B2's own repair variant in its search space --
otherwise B4 would not be a true upper bound over B2. Cheap here because only one dimension
is ever degraded at a time (at most 3 x 4 = 12 combinations).
"""
from hybriddqr.quality_diagnosis import DIMENSIONS
from hybriddqr.repair import apply_repair
from hybriddqr.robust_model_selection import MODEL_REGISTRY, run_model, top_ranked_model

DEFAULT_MODEL_KEY = {"classification": "linear", "regression": "linear", "clustering": "centroid"}


def run_b1_no_repair_default_model(df_polluted, metadata_entry, task) -> float:
    return run_model(task, DEFAULT_MODEL_KEY[task], df_polluted, metadata_entry)


def run_b2_full_repair_default_model(df_polluted, metadata_entry, task) -> float:
    df = df_polluted
    for dimension in DIMENSIONS:
        try:
            df = apply_repair(dimension, df, metadata_entry, task)
        except Exception:
            continue  # a repair operator that doesn't apply to this dataset shape is a no-op
    return run_model(task, DEFAULT_MODEL_KEY[task], df, metadata_entry)


def run_b3_no_repair_best_robust_model(df_polluted, metadata_entry, task, active_dimension) -> float:
    best_model_key = top_ranked_model(task, active_dimension)
    return run_model(task, best_model_key, df_polluted, metadata_entry)


def run_b4_oracle(df_polluted, metadata_entry, task, active_dimension) -> float:
    """Brute-force over every repair variant this module can produce (unrepaired,
    single-dimension repair of the one actually-degraded dimension, and B2's full
    all-6-dimension repair) x every candidate model. Must include the same repair variant
    B2 uses -- otherwise B4 wouldn't be a true oracle upper bound over B2's own action."""
    active_only_repaired = apply_repair(active_dimension, df_polluted, metadata_entry, task)
    fully_repaired = df_polluted
    for dimension in DIMENSIONS:
        try:
            fully_repaired = apply_repair(dimension, fully_repaired, metadata_entry, task)
        except Exception:
            continue

    best_metric = None
    for candidate_df in (df_polluted, active_only_repaired, fully_repaired):
        for model_key in MODEL_REGISTRY[task]:
            try:
                metric = run_model(task, model_key, candidate_df, metadata_entry)
            except Exception:
                continue
            if best_metric is None or metric > best_metric:
                best_metric = metric
    return best_metric if best_metric is not None else float("nan")


def run_all_baselines(df_polluted, metadata_entry, task, active_dimension) -> dict:
    return {
        "B1_no_repair_default": run_b1_no_repair_default_model(df_polluted, metadata_entry, task),
        "B2_full_repair_default": run_b2_full_repair_default_model(df_polluted, metadata_entry, task),
        "B3_no_repair_best_robust": run_b3_no_repair_best_robust_model(
            df_polluted, metadata_entry, task, active_dimension
        ),
        "B4_oracle": run_b4_oracle(df_polluted, metadata_entry, task, active_dimension),
    }
