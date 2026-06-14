"""
Unit tests for the six quality-diagnosis formulas HybridDQR's QAD module reuses from
DQ4AI's `polluters/*.py`. Each test builds a small, hand-computable dataframe -- several of
them are the exact worked examples from the DQ4AI paper (Section 3) -- and checks that
`Polluter.compute_quality_measure` returns the value the paper's own definition predicts.
This does not test HybridDQR-original code so much as it confirms the assumption
`quality_diagnosis.py` is built on: that DQ4AI's own formulas can be called directly and
trusted as-is.

Run with:
    python -m unittest hybriddqr.tests.test_quality_diagnosis -v
"""
import unittest

import pandas as pd

from polluters import (
    ConsistentRepresentationPolluter,
    CompletenessPolluter,
    FeatureAccuracyPolluter,
    TargetAccuracyPolluter,
    UniquenessPolluter,
    ClassBalancePolluter,
)
from hybriddqr.quality_diagnosis import scalarize_quality_measure


class TestConsistency(unittest.TestCase):
    def test_city_example_from_paper(self):
        # DQ4AI paper, Section 3.1 "Example": ten City values, 5 replacements needed ->
        # InCons(City) = 0.5 -> Consistency(d) = 0.5 (single-feature dataset).
        city = ["New York", "NYC", "Los Angeles", "LA", "San Francisco", "SF",
                "NY", "Los Angeles", "SF", "San Francisco"]
        df = pd.DataFrame({"City": city, "target": [0] * 10})

        polluter = ConsistentRepresentationPolluter(
            random_seed=42, percentage_polluted_rows=0.5, num_pollutable_columns=1,
            number_of_representations={"City": {"NYC": 1, "NY": 1}},
        )
        polluter.new_representations = {
            "City": {"New York": ["NYC", "NY"], "Los Angeles": ["LA"], "San Francisco": ["SF"]}
        }

        overall, pollutable_only = polluter.compute_quality_measure(df, df)
        self.assertAlmostEqual(overall, 0.5, places=6)
        self.assertAlmostEqual(pollutable_only, 0.5, places=6)
        self.assertAlmostEqual(scalarize_quality_measure("consistency", (overall, pollutable_only)), 0.5)


class TestCompleteness(unittest.TestCase):
    def test_two_feature_example_from_paper(self):
        # DQ4AI paper, Section 3.2 "Example": 2 features x 4 samples, 2 missing per feature
        # -> Completeness(d) = 1 - 0.5 = 0.5.
        df = pd.DataFrame({
            "target": [0, 0, 0, 0],
            "f1": [1, 2, -1, -1],
            "f2": [-1, -1, 3, 4],
        })
        polluter = CompletenessPolluter(
            pollution_percentages=0.0, target_feature="target",
            placeholder_numerical=-1, placeholder_categorical="empty",
            numerical_cols=["f1", "f2"], categorical_cols=[],
            random_seed=42,
        )
        quality = polluter.compute_quality_measure(df)
        self.assertAlmostEqual(quality, 0.5, places=6)


class TestUniqueness(unittest.TestCase):
    def test_ten_samples_three_duplicates_from_paper(self):
        # DQ4AI paper, Section 3.5 "Example": 10 samples, 3 exact duplicates (7 unique)
        # -> Uniqueness(d) = (7-1)/(10-1) = 2/3.
        rows = ["A", "B", "C", "D", "E", "F", "G", "A", "B", "C"]
        df = pd.DataFrame({"value": rows})
        self.assertEqual(len(df.drop_duplicates()), 7)

        polluter = UniquenessPolluter(
            duplicate_factor=1.0, distribution_function_name="same",
            distribution_function_parameters={}, target_feature="value",
            random_seed=42,
        )
        quality = polluter.compute_quality_measure(df)
        self.assertAlmostEqual(quality, 2 / 3, places=6)


class TestClassBalance(unittest.TestCase):
    def test_three_classes_paper_vs_actual_implementation(self):
        # DQ4AI paper, Section 3.6, Definition 6: ImBalance(d) = (1/2) * sum_{i,j} |n_i - n_j|.
        # Its own worked example (classes of size 10/20/30) computes
        # ImBalance = 0.5*(|10-20|+|10-30|+|20-30|) = 0.5*40 = 20, worst case = 60,
        # giving Balance(d) = 1 - 20/60 = 2/3 as stated in the paper text.
        #
        # The shipped implementation (polluters/classbalance.py) does NOT apply that 1/2
        # factor: it sums each unordered class pair exactly once (40, not 20) and divides
        # directly by the same worst-case denominator (60), giving 1 - 40/60 = 1/3 -- a
        # factor of two off from the paper's own stated example. Since HybridDQR's QAD module
        # calls this function directly (see quality_diagnosis.py), what matters for this
        # project is that our code matches DQ4AI's *actual* implementation, which is what
        # every DQ4AI experiment (and everything built on top of it here) really runs. We
        # assert the value the code produces and record the discrepancy with the paper text
        # here rather than silently relying on an assumption that turned out to be wrong.
        df = pd.DataFrame({
            "target": ["expensive"] * 10 + ["moderate"] * 20 + ["cheap"] * 30
        })
        polluter = ClassBalancePolluter(
            imbalance_level=0.0, target_column="target", n_samples=60, random_seed=42
        )
        quality = polluter.compute_quality_measure(df)
        self.assertAlmostEqual(quality, 1 / 3, places=6)


class TestFeatureAccuracy(unittest.TestCase):
    def test_categorical_example_from_paper(self):
        # DQ4AI paper, Section 3.3 "Example": 5 samples, City column, 1 mismatch
        # (LA should read Los Angeles) -> cFAccuracy(d) = 1 - 1/5 = 0.8.
        clean = pd.DataFrame({"City": ["New York", "Los Angeles", "San Francisco", "Chicago", "Boston"]})
        polluted = clean.copy()
        polluted.loc[1, "City"] = "LA"

        polluter = FeatureAccuracyPolluter(
            pollution_levels=0.0, categorical_cols=["City"], numerical_cols=[], random_seed=42
        )
        cat_quality, num_quality = polluter.compute_quality_measure(polluted, clean)
        self.assertAlmostEqual(cat_quality, 0.8, places=6)
        self.assertIsNone(num_quality)
        self.assertAlmostEqual(scalarize_quality_measure("feature_accuracy", (cat_quality, num_quality)), 0.8)

    def test_numerical_formula_self_consistency(self):
        # The paper's worked example only covers the categorical case; this checks the
        # numerical branch (Eq. 4/5) against an independently hand-computed value.
        clean = pd.DataFrame({"temp": [10.0, 20.0, 30.0]})
        polluted = pd.DataFrame({"temp": [10.0, 25.0, 30.0]})
        expected = 1 - ((0 + 5 + 0) / 3) / 20.0  # avg_dist / mean(clean) = 1.6667/20

        polluter = FeatureAccuracyPolluter(
            pollution_levels=0.0, categorical_cols=[], numerical_cols=["temp"], random_seed=42
        )
        cat_quality, num_quality = polluter.compute_quality_measure(polluted, clean)
        self.assertIsNone(cat_quality)
        self.assertAlmostEqual(num_quality, expected, places=6)


class TestTargetAccuracy(unittest.TestCase):
    def test_categorical_target_analogous_to_paper_example(self):
        # Same structure as the feature-accuracy example (Section 3.4: "calculated
        # similarly to feature accuracy"): 5 samples, 1 mismatch -> accuracy = 1 - 1/5 = 0.8.
        clean = pd.DataFrame({"label": ["yes", "no", "yes", "no", "yes"]})
        polluted = clean.copy()
        polluted.loc[2, "label"] = "no"

        polluter = TargetAccuracyPolluter(
            pollution_level=0.0, target_col="label", is_categorical=True, random_seed=42
        )
        quality = polluter.compute_quality_measure(polluted, clean)
        self.assertAlmostEqual(quality, 0.8, places=6)


if __name__ == "__main__":
    unittest.main()
