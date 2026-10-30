"""Small invariant tests for the statistics most vulnerable to transcription errors."""
import unittest

import numpy as np
from scipy import stats

from src.run_all import holm, required_n


class AnalysisTests(unittest.TestCase):
    def test_oc_rule_accepts_at_most_three(self):
        self.assertAlmostEqual(stats.binom.cdf(3, 60, 0.02), 0.9678058294539199)
        self.assertAlmostEqual(1 - stats.binom.cdf(3, 60, 0.05), 0.35271898839866056)
        self.assertEqual(stats.binom.cdf(3, 60, 0), 1)

    def test_holm_preserves_order_and_monotonicity(self):
        p = [0.04, 0.001, 0.02, 0.8]
        adjusted = holm(p)
        self.assertEqual(len(adjusted), len(p))
        self.assertTrue(all(x >= y for x, y in zip(adjusted, p)))
        order = np.argsort(p)
        self.assertTrue(np.all(np.diff(np.array(adjusted)[order]) >= 0))

    def test_power_result_is_minimum_integer(self):
        d = 0.37628544524557933
        n = required_n(d)
        self.assertEqual(n, 186)
        def power(size):
            df = 2*size-2
            critical = stats.t.ppf(1-.01/2, df)
            nc = abs(d)*np.sqrt(size/2)
            return stats.nct.sf(critical, df, nc)+stats.nct.cdf(-critical, df, nc)
        self.assertLess(power(n-1), .85)
        self.assertGreaterEqual(power(n), .85)


if __name__ == "__main__":
    unittest.main()
