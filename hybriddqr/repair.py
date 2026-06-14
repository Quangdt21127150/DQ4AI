"""
Module 2 of HybridDQR: Selective Repair (SR).

One function per quality dimension (Table 3 of the paper), each `(df, metadata_entry) ->
repaired_df`. `apply_repair(dimension, ...)` dispatches to the right one. The *decision* of
which dimensions to actually repair (cost-benefit gated) lives in `policy.py`, not here --
this module only implements the mechanics of each repair operator, unconditionally.

Windows note: cleanlab's `find_label_issues` and sklearn's `cross_val_predict` use
`multiprocessing.Pool` internally, which on Windows requires the calling script to guard its
entry point with `if __name__ == "__main__":` (spawn-based process creation re-imports the
main module). Every call here is additionally pinned to `n_jobs=1` to keep repair cheap and
avoid relying on multiprocessing at all in a module that may be invoked many times per
pollution level.
"""
import difflib
import warnings

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (registers IterativeImputer)
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_predict


def _placeholder_for_column(placeholders: dict, kind: str, column: str):
    """placeholders[kind] is either a single value shared by all columns of that kind, or a
    per-column dict (as used for house_prices_prepared.csv in metadata.json)."""
    value = placeholders.get(kind)
    if isinstance(value, dict):
        return value.get(column)
    return value


def repair_consistency(df: pd.DataFrame, metadata_entry: dict, similarity_threshold: float = 0.9) -> pd.DataFrame:
    """Entity normalization / string unification (Table 3). For each categorical column,
    cluster distinct values by string similarity and remap every minority-frequency variant
    within a cluster to the cluster's most-frequent ("canonical") value. Unsupervised --
    never looks at a clean reference.

    This correctly reverses DQ4AI's own pollution pattern for this dimension, which appends
    a trailing incrementing number to a string value (e.g. "SectorA" -> "SectorA1"), since
    those variants have high edit-distance similarity to the canonical value. It cannot
    recognise free-form semantic abbreviations (e.g. "NYC" for "New York"), which have low
    character-level similarity to the value they stand for -- see
    hybriddqr/tests/test_repair.py for a test documenting this limitation."""
    df = df.copy()
    for column in metadata_entry.get("categorical_cols", []):
        if column not in df.columns or column == metadata_entry.get("target"):
            continue
        values = df[column].astype(str)
        counts = values.value_counts()
        uniques = list(counts.index)
        canonical_for = {}
        assigned = set()
        for i, value in enumerate(uniques):
            if value in assigned:
                continue
            cluster = [value]
            for other in uniques[i + 1:]:
                if other in assigned:
                    continue
                ratio = difflib.SequenceMatcher(None, value, other).ratio()
                if ratio >= similarity_threshold:
                    cluster.append(other)
            canonical = max(cluster, key=lambda v: counts[v])
            for member in cluster:
                canonical_for[member] = canonical
                assigned.add(member)
        df[column] = values.map(canonical_for)
    return df


def repair_completeness(df: pd.DataFrame, metadata_entry: dict) -> pd.DataFrame:
    """Iterative imputation (MissForest-equivalent via IterativeImputer + a tree-based
    estimator) for numerical columns, most-frequent-value imputation for categorical
    columns (Table 3)."""
    df = df.copy()
    placeholders = metadata_entry.get("placeholders", {})
    numerical_cols = [c for c in metadata_entry.get("numerical_cols", []) if c in df.columns]
    categorical_cols = [c for c in metadata_entry.get("categorical_cols", []) if c in df.columns]

    for column in numerical_cols:
        placeholder = _placeholder_for_column(placeholders, "numerical", column)
        if placeholder is not None:
            df[column] = df[column].replace(placeholder, np.nan)
    for column in categorical_cols:
        placeholder = _placeholder_for_column(placeholders, "categorical", column)
        if placeholder is not None:
            df[column] = df[column].replace(placeholder, np.nan)

    if numerical_cols and df[numerical_cols].isna().any().any():
        imputer = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=20, random_state=42, n_jobs=1),
                                    random_state=42, max_iter=5)
        df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

    for column in categorical_cols:
        if df[column].isna().any():
            mode = df[column].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "empty"
            df[column] = df[column].fillna(fill_value)

    return df


def repair_feature_accuracy(df: pd.DataFrame, metadata_entry: dict, z_threshold: float = 3.0) -> pd.DataFrame:
    """Z-score outlier correction / winsorization (Table 3): clip numerical columns beyond
    +/- z_threshold standard deviations back to the threshold bound."""
    df = df.copy()
    for column in metadata_entry.get("numerical_cols", []):
        if column not in df.columns:
            continue
        col = df[column].astype(float)
        mean, std = col.mean(), col.std(ddof=0)
        if not std or np.isnan(std):
            continue
        z_scores = (col - mean) / std
        lower, upper = mean - z_threshold * std, mean + z_threshold * std
        df[column] = col.where(z_scores.abs() <= z_threshold, col.clip(lower=lower, upper=upper))
    return df


def repair_target_accuracy(df: pd.DataFrame, metadata_entry: dict, task: str) -> pd.DataFrame:
    """Confident learning (Table 3) for classification -- flag likely-mislabeled rows from
    out-of-fold predicted probabilities and relabel them to the model's prediction. For
    regression there is no analogous label-noise detector; a quick robust reference
    regressor's large-residual rows are corrected toward its prediction instead. For
    clustering this is a no-op: per DQ4AI's own finding (Section 6.3), target-accuracy
    pollution has no effect on clustering algorithms because they never see labels during
    fitting -- the target column only exists for post-hoc AMI evaluation, and "repairing"
    an evaluation-only ground-truth column from itself would be circular."""
    if task == "clustering":
        return df

    df = df.copy()
    target = metadata_entry["target"]
    feature_cols = [c for c in df.columns if c != target]
    X = pd.get_dummies(df[feature_cols], drop_first=True)

    if task == "classification":
        from cleanlab.filter import find_label_issues

        y = df[target]
        if y.nunique() < 2 or len(df) < 10:
            return df
        try:
            cv = min(5, y.value_counts().min(), len(df) // 2)
            cv = max(cv, 2)
            pred_probs = cross_val_predict(
                LogisticRegression(max_iter=1000), X, y, cv=cv, method="predict_proba", n_jobs=1
            )
            issues = find_label_issues(labels=y.values, pred_probs=pred_probs, n_jobs=1)
            predicted_labels = pred_probs.argmax(axis=1)
            classes = np.unique(y.values)
            df.loc[issues, target] = classes[predicted_labels[issues]]
        except Exception as exc:  # pragma: no cover - defensive: never let repair crash the pipeline
            warnings.warn(f"repair_target_accuracy (classification) skipped: {exc}")
        return df

    # regression
    y = df[target].astype(float)
    try:
        preds = cross_val_predict(LinearRegression(), X, y, cv=5, n_jobs=1)
        residuals = (y - preds).abs()
        threshold = residuals.mean() + 3 * residuals.std(ddof=0)
        outliers = residuals > threshold
        df.loc[outliers, target] = preds[outliers.values]
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"repair_target_accuracy (regression) skipped: {exc}")
    return df


def repair_uniqueness(df: pd.DataFrame, metadata_entry: dict) -> pd.DataFrame:
    """Hash-based exact-duplicate removal (Table 3) -- same operation already used in
    polluters/uniqueness_polluter.py, just applied as a correction rather than for
    pollution bookkeeping."""
    return df.drop_duplicates().reset_index(drop=True)


def repair_class_balance(df: pd.DataFrame, metadata_entry: dict) -> pd.DataFrame:
    """SMOTE / undersampling (Table 3) to rebalance the target classes. Uses SMOTENC when
    categorical features are present (mixed-type data), plain SMOTE for all-numeric
    features, falling back to random oversampling if SMOTE's k-neighbors requirement can't
    be met by the smallest class."""
    from imblearn.over_sampling import SMOTE, SMOTENC, RandomOverSampler

    target = metadata_entry["target"]
    feature_cols = [c for c in df.columns if c != target]
    categorical_cols = [c for c in metadata_entry.get("categorical_cols", []) if c in feature_cols]
    X, y = df[feature_cols], df[target]

    min_class_count = y.value_counts().min()
    if min_class_count < 2 or y.nunique() < 2:
        return df

    k_neighbors = max(1, min(5, min_class_count - 1))
    try:
        if categorical_cols:
            cat_indices = [feature_cols.index(c) for c in categorical_cols]
            sampler = SMOTENC(categorical_features=cat_indices, random_state=42, k_neighbors=k_neighbors)
        else:
            sampler = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_res, y_res = sampler.fit_resample(X, y)
    except Exception:
        X_res, y_res = RandomOverSampler(random_state=42).fit_resample(X, y)

    resampled = X_res.copy()
    resampled[target] = y_res
    return resampled.reset_index(drop=True)


REPAIR_FUNCTIONS = {
    "consistency": lambda df, metadata_entry, task: repair_consistency(df, metadata_entry),
    "completeness": lambda df, metadata_entry, task: repair_completeness(df, metadata_entry),
    "feature_accuracy": lambda df, metadata_entry, task: repair_feature_accuracy(df, metadata_entry),
    "target_accuracy": lambda df, metadata_entry, task: repair_target_accuracy(df, metadata_entry, task),
    "uniqueness": lambda df, metadata_entry, task: repair_uniqueness(df, metadata_entry),
    "class_balance": lambda df, metadata_entry, task: repair_class_balance(df, metadata_entry),
}


def apply_repair(dimension: str, df: pd.DataFrame, metadata_entry: dict, task: str) -> pd.DataFrame:
    """Dispatch to the repair operator for `dimension`. `task` in {'classification',
    'regression', 'clustering'} -- only used by target_accuracy (repair strategy differs)
    and class_balance (skipped for clustering, which has no train/test target semantics
    the same way, see policy.py's SEVERITY_THRESHOLDS usage)."""
    return REPAIR_FUNCTIONS[dimension](df, metadata_entry, task)
