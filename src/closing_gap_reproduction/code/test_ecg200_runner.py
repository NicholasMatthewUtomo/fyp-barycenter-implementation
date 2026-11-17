"""Check class selection and validation for both ECG200 candidate rules."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_ecg200 import select_class_indices, validate_result
import wb_bary_algorithms as wb


class ECGRunnerTests(unittest.TestCase):
    def test_ten_inputs_include_five_from_each_split_and_only_requested_label(self):
        labels = np.tile([1, -1], 16)
        splits = np.repeat(["train", "test"], 16)
        for label in (1, -1):
            selected = select_class_indices(labels, splits, label, 10)
            self.assertEqual(len(np.unique(selected)), 10)
            np.testing.assert_array_equal(labels[selected], np.full(10, label))
            self.assertEqual(int((splits[selected] == "train").sum()), 5)
            self.assertEqual(int((splits[selected] == "test").sum()), 5)
            np.testing.assert_array_equal(selected[:2], [0, 16] if label == 1 else [1, 17])
            np.testing.assert_array_equal(selected, select_class_indices(labels, splits, label, 10))

    def test_selection_uses_remaining_split_when_one_is_exhausted(self):
        labels = np.ones(5, dtype=int)
        splits = np.array(["train", "test", "test", "test", "test"])
        np.testing.assert_array_equal(select_class_indices(labels, splits, 1, 5), np.arange(5))
        np.testing.assert_array_equal(select_class_indices(labels, splits, 1, None), np.arange(5))
        for k in (0, 6):
            with self.subTest(k=k), self.assertRaises(ValueError):
                select_class_indices(labels, splits, 1, k)

    def test_validation_accepts_t1_grid_and_rejects_wrong_support(self):
        measures = [wb.Measure([[0, 0], [1, 2]], [0.5, 0.5])]
        candidates, _ = wb.build_W1_candidates_subspace_enumeration(measures, t=1)
        result = dict(S=candidates, z=np.full(len(candidates), 1 / len(candidates)), objective=1.0)
        self.assertEqual(validate_result(result, candidates)["support_size"], 4)
        with self.assertRaises(ValueError):
            validate_result(result, wb._union_support_points(measures))
        pooled = wb._union_support_points(measures)
        result = dict(S=pooled, z=np.full(len(pooled), 1 / len(pooled)), objective=0.0)
        self.assertEqual(validate_result(result, pooled)["support_size"], 2)


if __name__ == "__main__":
    unittest.main()
