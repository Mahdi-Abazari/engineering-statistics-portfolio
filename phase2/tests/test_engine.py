"""Meaningful checks of the custom estimators and interval logic."""
import unittest

import numpy as np
from scipy import stats

from src.estimation import theta_mom,theta_mle,theta_mle_numeric_verify,theta_exact_ci
from src.regression import (ols_fit,ols_linregress_verify,regression_mle_numeric_verify,
                            coefficient_inference,response_intervals)


class EstimationTests(unittest.TestCase):
    def test_direct_mom_mle_and_independent_optimizer(self):
        sample=np.array([1.,2.,3.,4.,8.])
        self.assertEqual(theta_mom(sample),3.6)
        self.assertEqual(theta_mle(sample),3.6)
        self.assertAlmostEqual(theta_mle_numeric_verify(sample)["estimate"],3.6,places=6)

    def test_exact_interval_orders_endpoints(self):
        lower,upper=theta_exact_ci(356.95,60)
        self.assertLess(lower,356.95)
        self.assertGreater(upper,356.95)
        self.assertAlmostEqual(lower,281.4112427394513,places=6)


class RegressionTests(unittest.TestCase):
    def test_custom_fit_matches_independent_library(self):
        x=np.array([1.,2.,3.,4.,5.,6.])
        y=np.array([3.,5.,5.5,8.,9.,10.5])
        custom=ols_fit(x,y)
        library=ols_linregress_verify(x,y)
        numeric=regression_mle_numeric_verify(x,y)
        for key in ["a","b"]:
            self.assertAlmostEqual(custom[key],library[key],places=8)
            self.assertAlmostEqual(custom[key],numeric[key],places=5)

    def test_prediction_interval_exceeds_mean_interval(self):
        x=np.array([1.,2.,3.,4.,5.,6.])
        y=np.array([3.,5.,5.5,8.,9.,10.5])
        fit=ols_fit(x,y)
        infer=coefficient_inference(fit)
        prediction=response_intervals(fit,4.)
        self.assertLess(infer["slope_p"],.05)
        mean_width=np.diff(prediction["mean_ci"])[0]
        new_width=np.diff(prediction["prediction_interval"])[0]
        self.assertGreater(new_width,mean_width)


if __name__=="__main__":
    unittest.main()
