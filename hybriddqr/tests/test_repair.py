"""
Repair-effectiveness tests: for each of the six operators in `hybriddqr/repair.py`, build a
small dataframe with a known, injected quality problem and check that the repair actually
fixes what it claims to fix. These are deliberately simple, direct checks (duplicate counts,
missing-value counts, class-count spread, etc.) rather than round-tripping through a full
Polluter/metadata setup for every dimension -- that round trip is already covered, for the
dimensions where it applies, by experiment_runner.py's real (dataset, dimension, pollution
level) sweep and by test_quality_diagnosis.py's formula checks.

Run with:
    python -m unittest hybriddqr.tests.test_repair -v
"""
import unittest

import numpy as np
import pandas as pd

from hybriddqr.repair import (
    repair_consistency,
    repair_completeness,
    repair_feature_accuracy,
    repair_target_accuracy,
    repair_uniqueness,
    repair_class_balance,
)


class TestRepairConsistency(unittest.TestCase):
    def test_trailing_suffix_variants_collapse_to_majority_representation(self):
        # DQ4AI's own ConsistentRepresentationPolluter injects variants by appending a
        # trailing incrementing number to string values (see
        # `_generate_new_representations` in consistent_representation_polluter.py), e.g.
        # "SectorA" -> "SectorA1", "SectorA2". This is the actual shape of polluted data
        # repair_consistency has to handle in the real experiment sweep, so the test injects
        # that pattern rather than a free-form abbreviation.
        sector = ["SectorA"] * 6 + ["SectorA1"] * 2 + ["SectorA2"] * 1 + ["SectorB"] * 4 + ["SectorB1"] * 1
        df = pd.DataFrame({"sector": sector, "target": [0] * len(sector)})
        metadata_entry = {"categorical_cols": ["sector"], "target": "target"}

        before_unique = df["sector"].nunique()
        repaired = repair_consistency(df, metadata_entry)
        after_unique = repaired["sector"].nunique()

        self.assertEqual(before_unique, 5)  # SectorA, SectorA1, SectorA2, SectorB, SectorB1
        self.assertLess(after_unique, before_unique)
        self.assertEqual(set(repaired["sector"].unique()), {"SectorA", "SectorB"})

    def test_semantic_abbreviations_are_a_known_limitation(self):
        # By contrast, a free-form abbreviation such as "NYC" for "New York" has low
        # character-level similarity to the canonical value, so the edit-distance-based
        # matching in repair_consistency cannot recognise it as the same entity. This test
        # documents that limitation rather than hiding it: fixing semantic abbreviations
        # would need a lookup table or embedding-based matching, which is out of scope here.
        city = ["New York"] * 6 + ["NYC"] * 2 + ["Los Angeles"] * 4
        df = pd.DataFrame({"City": city, "target": [0] * len(city)})
        metadata_entry = {"categorical_cols": ["City"], "target": "target"}

        repaired = repair_consistency(df, metadata_entry)
        self.assertIn("NYC", repaired["City"].unique())


class TestRepairCompleteness(unittest.TestCase):
    def test_missing_values_are_imputed(self):
        df = pd.DataFrame({
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
            "f1": [1.0, 2.0, -1.0, 4.0, 5.0, -1.0, 7.0, 8.0],
            "f2": ["a", "empty", "a", "b", "empty", "b", "a", "b"],
        })
        metadata_entry = {
            "target": "target", "numerical_cols": ["f1"], "categorical_cols": ["f2"],
            "placeholders": {"numerical": -1, "categorical": "empty"},
        }
        n_missing_before = (df["f1"] == -1).sum() + (df["f2"] == "empty").sum()
        repaired = repair_completeness(df, metadata_entry)
        n_missing_after = (repaired["f1"] == -1).sum() + (repaired["f2"] == "empty").sum()

        self.assertEqual(n_missing_before, 4)
        self.assertEqual(n_missing_after, 0)


class TestRepairFeatureAccuracy(unittest.TestCase):
    def test_outliers_are_winsorized(self):
        rng = np.random.default_rng(0)
        values = rng.normal(loc=50, scale=1, size=200).tolist()
        values[0] = 5000.0  # an obvious outlier well beyond 3 sigma
        df = pd.DataFrame({"f1": values})
        metadata_entry = {"numerical_cols": ["f1"]}

        repaired = repair_feature_accuracy(df, metadata_entry)
        mean, std = df["f1"].mean(), df["f1"].std(ddof=0)
        z_before = abs((df.loc[0, "f1"] - mean) / std)
        z_after = abs((repaired.loc[0, "f1"] - mean) / std)

        self.assertGreater(z_before, 3.0)
        self.assertLessEqual(z_after, 3.0 + 1e-6)


class TestRepairUniqueness(unittest.TestCase):
    def test_exact_duplicates_are_removed(self):
        df = pd.DataFrame({"a": [1, 2, 3, 1, 2], "b": ["x", "y", "z", "x", "y"]})
        self.assertEqual(df.duplicated().sum(), 2)
        repaired = repair_uniqueness(df, metadata_entry={})
        self.assertEqual(repaired.duplicated().sum(), 0)
        self.assertEqual(len(repaired), 3)


class TestRepairClassBalance(unittest.TestCase):
    def test_minority_class_is_oversampled(self):
        rng = np.random.default_rng(0)
        n_majority, n_minority = 90, 10
        df = pd.DataFrame({
            "f1": rng.normal(size=n_majority + n_minority),
            "f2": rng.normal(size=n_majority + n_minority),
            "target": [0] * n_majority + [1] * n_minority,
        })
        metadata_entry = {"target": "target", "categorical_cols": []}

        repaired = repair_class_balance(df, metadata_entry)
        counts = repaired["target"].value_counts()
        self.assertEqual(counts[0], counts[1])
        self.assertGreater(len(repaired), len(df))


class TestRepairTargetAccuracy(unittest.TestCase):
    def test_classification_flags_and_corrects_some_flipped_labels(self):
        rng = np.random.default_rng(0)
        n = 200
        f1 = rng.normal(size=n)
        true_label = (f1 > 0).astype(int)
        noisy_label = true_label.copy()
        # Flip 15 labels that are far from the decision boundary -- unambiguous label noise
        # a reference classifier should be confident about.
        far_positive = np.argsort(-f1)[:8]
        far_negative = np.argsort(f1)[:7]
        noisy_label[far_positive] = 0
        noisy_label[far_negative] = 1

        df = pd.DataFrame({"f1": f1, "target": noisy_label})
        metadata_entry = {"target": "target"}

        n_wrong_before = (noisy_label != true_label).sum()
        repaired = repair_target_accuracy(df, metadata_entry, task="classification")
        n_wrong_after = (repaired["target"].values != true_label).sum()

        self.assertEqual(n_wrong_before, 15)
        self.assertLess(n_wrong_after, n_wrong_before)

    def test_clustering_is_a_no_op(self):
        df = pd.DataFrame({"f1": [1, 2, 3], "target": ["a", "b", "c"]})
        metadata_entry = {"target": "target"}
        repaired = repair_target_accuracy(df, metadata_entry, task="clustering")
        pd.testing.assert_frame_equal(df, repaired)


if __name__ == "__main__":
    unittest.main()
