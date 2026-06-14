# HybridDQR

A real, runnable implementation of **HybridDQR** -- the framework proposed in *"Hybrid Data
Quality Repair and Robust Modeling for Enhanced Machine Learning Performance on Polluted
Tabular Data"*. That paper explicitly ships **no empirical validation** ("we describe a
comprehensive experimental protocol **that will be used to validate the framework in
follow-up empirical work**"; its Table 7 is "**projected**... derived analytically", not
measured). This package is that follow-up: it builds the paper's three modules on top of
the DQ4AI benchmark already in this repo and runs the paper's own Section 8 protocol
(HybridDQR vs. four baselines) to produce its first real numbers.

This is a brand-new top-level package -- nothing in the repo's protected "Root code" commit
(`classification/`, `regression/`, `clustering/`, `polluters/`, `metadata.json`,
`requirements.txt`, ...) is modified. It imports and reuses that code directly.

## Why it reuses so much of DQ4AI

The paper adopts DQ4AI's own six data-quality dimensions verbatim ("We adopt the six data
quality dimensions defined in [DQ4AI]") and its own ten benchmark datasets/three ML tasks
("We use the same ten tabular datasets as the DQ4AI benchmark"). So rather than re-deriving
quality metrics or re-implementing yet another `LogisticRegression`/`RandomForestRegressor`
wrapper, this package calls straight into:

- `polluters/*.py` -- every dimension's exact, already-tested `compute_quality_measure`
  formula (Module 1, Quality-Aware Diagnosis, in `quality_diagnosis.py`).
- `classification/experiments.py`, `regression/experiments.py`, `clustering/experiments.py`
  -- the model classes used as HybridDQR's robust-model candidates (Module 3, in
  `robust_model_selection.py`).

## Module map (paper section -> file)

| Paper concept | File |
|---|---|
| Module 1: Quality-Aware Diagnosis (Def. 1) | `quality_diagnosis.py` |
| Module 2: Selective Repair (Table 3) | `repair.py` |
| Module 3: Robust Model Selection (Def. 3, Table 4, Eq. 3) | `robust_model_selection.py` |
| Hybrid Decision Policy (Def. 4) + Algorithm 1 | `policy.py` |
| Baselines B1-B4 (Section 8.3) | `baselines.py` |
| Experimental protocol (Section 8) | `experiment_runner.py` |

## Scope of this validation run (deliberately small)

The paper's own protocol sweeps 10 datasets x 6 dimensions x 0-100% pollution in 5-10%
steps x 5 random seeds x many candidate models per task -- a multi-hour-per-task grid even
with lightweight models, and considerably longer with deep-learning candidates in the mix.
This validation deliberately trades that breadth for a run that completes quickly and
reliably, while exercising every module's logic on real data:

- **One dataset per task**: `SouthGermanCredit.csv` (classification), `vw_prepared.csv`
  (regression), `bank_2967137.csv` (clustering, one of the five already-prepared samples).
- **~5 pollution levels per dimension** (one random seed), not the full sweep.
- **Lightweight, sklearn-backed candidate models only.** All `TabNet*`, `Pytorch*`, and
  `AutoencoderExperiment` classes are excluded: they add PyTorch training-loop overhead and
  third-party-library dependencies (TabNet, custom DataLoader-based models) that make
  runtime far less predictable than the sklearn estimators used elsewhere, which does not
  suit a diagnose-repair-select pipeline meant to run quickly and repeatedly. Regression has
  no genuinely-sklearn "Deep NN" candidate in this codebase (`MLPRegressionExperiment`'s
  name is shadowed by a custom `torch.nn.Module` class defined later in
  `regression/experiments.py`), so regression's robustness ranking uses 3 families
  (Ensemble/Linear/Tree) instead of 4.
- Every dataset is polluted **one dimension at a time** (DQ4AI's own methodology), so for
  any given run the other 5 dimensions are known-by-construction to be at quality 1.0 --
  `quality_diagnosis.py` never needs a from-scratch, ground-truth-free scoring path.

None of this changes the *logic* of HybridDQR being validated -- QAD, the repair operators,
WRS-based model selection, and the hybrid policy all run exactly as specified. It only
limits how much of the paper's dataset/pollution-level grid gets swept in one pass.

## Design decisions worth knowing about

- **Consistency's quality measure needs the `Polluter` instance itself**, not just a
  before/after dataframe pair -- `ConsistentRepresentationPolluter.compute_quality_measure`
  reads `self.new_representations`, built during `pollute()`. `experiment_runner.py`
  therefore always measures quality by calling `compute_quality_measure` on the *same*
  polluter instance that produced the pollution (both for the initial diagnosis and, after
  repair, for re-diagnosis) rather than reimplementing the formula.
- **Delta_i (Eq. 2, repair benefit estimate)** is measured empirically: the dimension's
  repair operator is actually applied once, and a cheap reference model's performance
  before/after gives Delta_i, rather than an analytical estimate. The reference model is
  "linear" for classification/regression (paper: "e.g., a linear classifier") and KMeans
  for clustering (no supervised reference exists there).
- **theta (target performance)** defaults to 90% of the best clean-data performance across
  all of a task's candidate models, computed once per dataset before the pollution sweep.
- **B2 (full repair) intentionally repairs all 6 dimensions unconditionally**, including the
  5 that are already at quality 1.0 in this single-active-dimension setup. That's not
  wasted computation -- it's the exact dynamic the paper motivates HybridDQR against:
  blanket repair can spend cost (and risk distorting already-clean data) where a
  cost-aware policy would correctly skip.
- **Windows multiprocessing**: `cleanlab.filter.find_label_issues` and
  `sklearn.model_selection.cross_val_predict` use `multiprocessing.Pool` internally, which on
  Windows re-imports the calling script as `__main__` when spawning workers. Every call here
  is pinned to `n_jobs=1`, and any entry-point script needing them is guarded with
  `if __name__ == "__main__":`, to avoid the spawn-storm/deadlock failure mode this pattern
  is prone to on Windows.

## Running it

```bash
# from the repo root, with the dq4ai conda env active
python -m hybriddqr.experiment_runner --task classification
python -m hybriddqr.experiment_runner --task regression
python -m hybriddqr.experiment_runner --task clustering
python -m hybriddqr.experiment_runner            # all three, sequentially
```

Results are written incrementally (crash-safe, same pattern as DQ4AI's own runners) to
`hybriddqr/re-results/hybriddqr_results_<task>.json`, one entry per (dimension, pollution
level) with the quality profile before/after repair, the policy's decision, the model WRS
selected, HybridDQR's metric, and all four baselines (B1-B4).

## Reading the results

For each task, F1 (classification) / R^2 (regression) / AMI (clustering) is reported for
HybridDQR and all four baselines at each pollution level. `hybriddqr_results_<task>.json`
records, per (dimension, pollution level): the quality profile before/after repair, the
policy's decision (skip/repair/robust/both) and its `delta_estimate`, the model WRS
selected, HybridDQR's metric, and all four baselines. `summary.json` aggregates these into
per-task averages and decision counts.

By construction, HybridDQR's metric is always <= B4 (the oracle's search space is a superset
of what HybridDQR can produce -- see `baselines.py`). It is **not** guaranteed to be >= B1:
in this validation, HybridDQR matched or beat B1 on 12/29 classification points, 23/29
regression points, and 29/29 clustering points. The classification shortfall is informative
rather than a bug -- averaged over all points, B2 (unconditional full repair) clearly
outperforms HybridDQR there (0.91 vs 0.66 mean F1), which means the default severity
thresholds `tau_i` (Table 2 of the paper) are, on this dataset, conservative enough to
leave real repair opportunities on the table as `robust`/`skip` decisions. Clustering shows
the opposite and arguably more paper-relevant pattern: B2 there is *worse* than B1
(0.011 vs 0.020 mean AMI), i.e. unconditional repair actively hurts, which is exactly the
failure mode HybridDQR's cost-aware policy is designed to avoid.
