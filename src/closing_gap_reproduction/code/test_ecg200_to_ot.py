"""Check ECG support thinning without changing time, amplitudes, or labels."""
import unittest

import numpy as np

from ecg200_to_ot import ecg_support_indices, waveforms_to_measures


class ECGMeasureTests(unittest.TestCase):
    def test_twelve_points_preserve_original_time_and_amplitude(self):
        waveforms = np.vstack((np.arange(96), -np.arange(96)))
        indices = ecg_support_indices(12)
        self.assertEqual(len(np.unique(indices)), 12)
        self.assertEqual((indices[0], indices[-1]), (0, 95))
        self.assertLessEqual(np.ptp(np.diff(indices)), 1)
        measures = waveforms_to_measures(waveforms, points=12)
        self.assertEqual(len(measures), 2)
        for original, measure in zip(waveforms, measures):
            self.assertEqual(measure.X.shape, (12, 2))
            np.testing.assert_allclose(measure.X[:, 0], indices / 95, atol=1e-15)
            np.testing.assert_array_equal(measure.X[:, 1], original[indices])
            np.testing.assert_array_equal(measure.w, np.full(12, 1 / 12))
            self.assertAlmostEqual(measure.w.sum(), 1)

    def test_default_retains_all_original_points(self):
        waveform = np.arange(96)[None, :]
        measure = waveforms_to_measures(waveform)[0]
        np.testing.assert_array_equal(measure.X[:, 0], np.linspace(0, 1, 96))
        np.testing.assert_array_equal(measure.X[:, 1], waveform[0])
        np.testing.assert_array_equal(measure.w, np.full(96, 1 / 96))

    def test_invalid_point_counts_are_rejected(self):
        for count in (0, 1, 97, 12.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                waveforms_to_measures(np.ones((1, 96)), points=count)


if __name__ == "__main__":
    unittest.main()
