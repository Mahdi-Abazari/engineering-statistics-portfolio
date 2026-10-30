"""Exponential scale estimation.

Reported estimators use base arithmetic. SciPy fitting lives only in *_verify
functions; distribution quantiles and generic optimization are allowed checks.
"""
from __future__ import annotations

import math
import numpy as np
from scipy import optimize, stats


def theta_mom(x: np.ndarray) -> float:
    """MoM: E[X]=theta implies theta_hat = sum(x_i)/n."""
    values = [float(v) for v in x]
    if not values or any(v <= 0 or not math.isfinite(v) for v in values):
        raise ValueError("Finite positive lifetimes are required")
    return sum(values) / len(values)


def theta_mle(x: np.ndarray) -> float:
    """Closed-form MLE: d log L/dtheta = -n/theta + sum(x)/theta² = 0."""
    return theta_mom(x)


def theta_mle_numeric_verify(x: np.ndarray) -> dict:
    """Independently minimize exponential negative log likelihood in log theta."""
    a = np.asarray(x, float)
    n, total = len(a), float(a.sum())
    def objective(z):
        theta = math.exp(float(z[0]))
        return n * math.log(theta) + total / theta
    def gradient(z):
        theta = math.exp(float(z[0]))
        return np.array([n - total / theta])
    result = optimize.minimize(objective, [math.log(float(np.median(a)))],
                               jac=gradient, method="BFGS", tol=1e-11)
    estimate = math.exp(float(result.x[0]))
    if abs(estimate - theta_mle(a)) > 1e-6:
        raise ArithmeticError("Numerical and closed-form MLE disagree")
    return {"estimate": estimate, "optimizer_success": bool(result.success),
            "gradient_norm": float(np.linalg.norm(result.jac))}


def theta_expon_fit_verify(x: np.ndarray) -> float:
    """Library-only verification; never used to construct reported estimates."""
    loc, scale = stats.expon.fit(np.asarray(x, float), floc=0)
    assert loc == 0
    return float(scale)


def theta_exact_ci(mean: float, n: int, alpha: float = .05) -> tuple[float, float]:
    """Invert 2*n*Xbar/theta ~ chi-square(2n) for an exact interval."""
    return (2*n*mean/stats.chi2.ppf(1-alpha/2, 2*n),
            2*n*mean/stats.chi2.ppf(alpha/2, 2*n))


def bootstrap_means(x: np.ndarray, b: int, rng: np.random.Generator) -> np.ndarray:
    """Draw b nonparametric resamples and calculate each sample mean."""
    a = np.asarray(x, float)
    indices = rng.integers(0, len(a), size=(b, len(a)))
    return a[indices].mean(axis=1)


def percentile_ci(draws: np.ndarray, alpha: float = .05) -> tuple[float, float]:
    """Central percentile interval from a bootstrap statistic distribution."""
    return tuple(float(v) for v in np.quantile(draws, [alpha/2, 1-alpha/2]))


def reliability(theta: float | np.ndarray, hours: float = 500) -> float | np.ndarray:
    """Exponential survival function R(t)=exp(-t/theta); MLE by invariance."""
    return np.exp(-hours / np.asarray(theta))


def mc_estimates(theta: float, n: int, trials: int,
                 rng: np.random.Generator, chunk: int = 500) -> np.ndarray:
    """Simulate sample-mean estimates in bounded-memory chunks."""
    estimates = np.empty(trials)
    for start in range(0, trials, chunk):
        stop = min(start + chunk, trials)
        estimates[start:stop] = rng.exponential(theta, size=(stop-start, n)).mean(axis=1)
    return estimates


def coverage_experiment(theta: float, n: int, trials: int, boot: int,
                        rng: np.random.Generator, batch: int = 40) -> dict:
    """Repeated-sampling coverage of exact and percentile-bootstrap 95% CIs."""
    exact_hits = bootstrap_hits = 0
    for start in range(0, trials, batch):
        count = min(batch, trials-start)
        samples = rng.exponential(theta, size=(count, n))
        means = samples.mean(axis=1)
        lower = 2*n*means/stats.chi2.ppf(.975, 2*n)
        upper = 2*n*means/stats.chi2.ppf(.025, 2*n)
        exact_hits += int(np.count_nonzero((lower <= theta) & (theta <= upper)))
        indices = rng.integers(0, n, size=(count, boot, n))
        boot_means = np.take_along_axis(samples[:, None, :], indices, axis=2).mean(axis=2)
        boot_ci = np.quantile(boot_means, [.025, .975], axis=1)
        bootstrap_hits += int(np.count_nonzero((boot_ci[0] <= theta) & (theta <= boot_ci[1])))
    return {"trials": trials, "bootstrap_resamples_per_trial": boot,
            "exact_coverage": exact_hits/trials, "bootstrap_coverage": bootstrap_hits/trials,
            "exact_hits": exact_hits, "bootstrap_hits": bootstrap_hits}
