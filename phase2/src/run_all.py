"""Reproduce all Phase 2 numerical results and publication-resolution figures."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from .estimation import (theta_mom, theta_mle, theta_mle_numeric_verify,
                         theta_expon_fit_verify, theta_exact_ci, bootstrap_means,
                         percentile_ci, reliability, mc_estimates, coverage_experiment)
from .regression import (ols_fit, ols_linregress_verify, regression_mle_numeric_verify,
                         coefficient_inference, response_intervals, residual_diagnostics,
                         regression_mc)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
OUT = ROOT / "reports" / "results.json"
SEED = 403109076
ALPHA = .05
M_BIAS = 10000
M_CONSISTENCY = 3000
B_DATA = 5000
M_COVERAGE = 2000
B_COVERAGE = 1000
M_REGRESSION = 10000
N_GRID = [10, 50, 200, 1000, 5000]
BLUE, ORANGE, GREEN = "#176B87", "#E29547", "#5C8A57"


def save(name: str) -> None:
    """Save the active figure as a crisp PNG."""
    FIG.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read and validate the supplied raw files without modifying them."""
    life = pd.read_csv(ROOT / "lifetimes.csv")
    process = pd.read_csv(ROOT / "process.csv")
    assert list(life.columns) == ["id", "lifetime_hours"] and len(life) == 60
    assert list(process.columns) == ["temp_C", "hardness_HV"] and len(process) == 40
    assert life.notna().all().all() and process.notna().all().all()
    assert life["id"].is_unique
    x = life["lifetime_hours"].to_numpy(float)
    temp = process["temp_C"].to_numpy(float)
    hardness = process["hardness_HV"].to_numpy(float)
    assert np.all(np.isfinite(x)) and np.all(x > 0)
    assert np.all(np.isfinite(temp)) and np.all(np.isfinite(hardness))
    return x, temp, hardness


def part_a(x: np.ndarray, rng: np.random.Generator) -> dict:
    """Execute A1-A8 with simulations and coverage checks."""
    n = len(x)
    theta = theta_mle(x)
    desc = {"n": n, "mean": float(x.mean()), "median": float(np.median(x)),
            "sample_sd": float(x.std(ddof=1)), "minimum": float(x.min()),
            "maximum": float(x.max()), "cv": float(x.std(ddof=1)/x.mean())}
    plt.hist(x, bins=16, density=True, color=BLUE, alpha=.62, edgecolor="white")
    xx = np.linspace(0, max(x)*1.08, 350)
    plt.plot(xx, stats.expon.pdf(xx, scale=theta), color=ORANGE, lw=2.5)
    plt.title("Failure times broadly follow an exponential shape")
    plt.xlabel("Lifetime to failure (hours)"); plt.ylabel("Density")
    save("01_lifetime_histogram")

    numeric = theta_mle_numeric_verify(x)
    scipy_check = theta_expon_fit_verify(x)
    if abs(theta-scipy_check) > 1e-10:
        raise AssertionError("Exponential library verification failed")
    estimates = mc_estimates(theta, n, M_BIAS, rng)
    theoretical_variance = theta**2/n
    empirical_variance = float(estimates.var(ddof=1))
    plt.hist(estimates, bins=45, density=True, color=BLUE, alpha=.66)
    plt.axvline(theta, color=ORANGE, lw=2.5, label="Generating theta")
    plt.title(f"Sample-mean estimator is centered near theta across {M_BIAS:,} trials")
    plt.xlabel("Estimated mean lifetime (hours)"); plt.ylabel("Density")
    plt.legend(); save("02_estimator_mc")

    grid = []
    largest = None
    for size in N_GRID:
        draws = mc_estimates(theta, size, M_CONSISTENCY, rng)
        grid.append({"n":size, "empirical_variance":float(draws.var(ddof=1)),
                     "theoretical_variance":theta**2/size,
                     "empirical_bias":float(draws.mean()-theta)})
        if size == N_GRID[-1]:
            largest = draws
    sizes = np.array(N_GRID)
    empirical = np.array([row["empirical_variance"] for row in grid])
    theoretical = theta**2/sizes
    plt.plot(sizes, empirical, "o-", color=BLUE, label="Monte Carlo")
    plt.plot(sizes, theoretical, "--", color=ORANGE, label=r"$\theta^2/n$")
    plt.title("Estimator variance approaches zero as sample size grows")
    plt.xlabel("Sample size n"); plt.ylabel("Variance of estimated lifetime (hours²)")
    plt.legend(); plt.grid(alpha=.2); save("03_consistency")
    plt.loglog(sizes, empirical, "o-", color=BLUE, label="Monte Carlo")
    plt.loglog(sizes, theoretical, "--", color=ORANGE, label="Theoretical slope -1")
    plt.title("Log-log variance follows the expected 1/n scaling")
    plt.xlabel("Sample size n (log scale)"); plt.ylabel("Estimator variance (log scale)")
    plt.legend(); plt.grid(alpha=.2); save("04_consistency_loglog")
    standardized = (largest-theta)/(theta/np.sqrt(N_GRID[-1]))
    stats.probplot(standardized, dist="norm", plot=plt)
    plt.title("At n = 5,000, standardized estimates are approximately normal")
    plt.xlabel("Theoretical standard-normal quantile"); plt.ylabel("Ordered standardized estimate")
    save("05_asymptotic_qq")

    plt.bar(["CRLB = theoretical variance", "Empirical variance"],
            [theoretical_variance, empirical_variance], color=[ORANGE, BLUE])
    plt.title("Observed variance is close to the Cramér-Rao bound")
    plt.ylabel("Variance of estimated lifetime (hours²)")
    save("06_efficiency")

    exact = theta_exact_ci(theta, n, ALPHA)
    boots = bootstrap_means(x, B_DATA, rng)
    boot_ci = percentile_ci(boots, ALPHA)
    plt.hist(boots, bins=45, density=True, color=BLUE, alpha=.65)
    plt.axvline(theta, color=ORANGE, lw=2, label="Observed estimate")
    plt.axvline(boot_ci[0], color=GREEN, ls="--", label="Bootstrap 95% limits")
    plt.axvline(boot_ci[1], color=GREEN, ls="--")
    plt.title("Bootstrap sampling distribution of mean lifetime")
    plt.xlabel("Bootstrap estimated lifetime (hours)"); plt.ylabel("Density")
    plt.legend(); save("07_bootstrap_lifetime")
    coverage = coverage_experiment(theta,n,M_COVERAGE,B_COVERAGE,rng)
    plt.bar(["Exact pivot", "Percentile bootstrap"],
            [coverage["exact_coverage"],coverage["bootstrap_coverage"]],
            color=[BLUE,ORANGE])
    plt.axhline(.95,color=GREEN,ls="--",lw=2,label="Nominal 95%")
    plt.ylim(.85,1); plt.ylabel("Empirical interval coverage")
    plt.title(f"Exact and bootstrap coverage across {M_COVERAGE:,} repeated samples")
    plt.legend(); save("08_coverage")

    rhat = float(reliability(theta))
    rboots = reliability(boots)
    rci = percentile_ci(rboots)
    ts = np.linspace(0,1000,250)
    plt.plot(ts,reliability(theta,ts),color=BLUE,lw=2.5)
    plt.scatter([500],[rhat],color=ORANGE,zorder=3)
    plt.annotate(f"R(500) = {rhat:.3f}",(500,rhat),xytext=(10,10),textcoords="offset points")
    plt.title("Estimated reliability decreases with operating time")
    plt.xlabel("Hours of operation"); plt.ylabel("Probability of survival")
    plt.ylim(0,1.02); save("09_reliability")

    return {"descriptive":desc, "theta_mom":theta_mom(x), "theta_mle":theta,
            "theta_numeric_verify":numeric, "theta_scipy_verify":scipy_check,
            "mc_bias_trials":M_BIAS, "mc_estimate_mean":float(estimates.mean()),
            "mc_bias":float(estimates.mean()-theta), "mc_variance":empirical_variance,
            "theoretical_variance":theoretical_variance, "crlb":theoretical_variance,
            "efficiency":theoretical_variance/empirical_variance,
            "consistency_trials_each":M_CONSISTENCY, "consistency":grid,
            "asymptotic_qq_r2":float(stats.probplot(standardized)[1][2]**2),
            "exact_ci":list(exact), "bootstrap_ci":list(boot_ci),
            "exact_width":exact[1]-exact[0], "bootstrap_width":boot_ci[1]-boot_ci[0],
            "data_bootstrap_resamples":B_DATA, "coverage":coverage,
            "reliability_500":rhat,"reliability_bootstrap_ci":list(rci)}


def part_b(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> dict:
    """Execute B1-B8 using the custom fit as the answer path."""
    fit = ols_fit(x,y)
    library = ols_linregress_verify(x,y)
    numerical = regression_mle_numeric_verify(x,y)
    for key in ["a","b"]:
        if abs(fit[key]-library[key]) > 1e-8 or abs(fit[key]-numerical[key]) > 1e-5:
            raise AssertionError(f"Regression verification failed for {key}")
    infer = coefficient_inference(fit)
    prediction = response_intervals(fit,400)
    diagnostics = residual_diagnostics(x,fit)
    mc, mc_slopes = regression_mc(x,fit,M_REGRESSION,rng)
    plt.hist(mc_slopes,bins=45,density=True,color=BLUE,alpha=.65)
    plt.axvline(fit["b"],color=ORANGE,lw=2.5,label="Generating slope")
    plt.title(f"Direct-sum slope estimates center near B over {M_REGRESSION:,} trials")
    plt.xlabel("Estimated slope (HV/°C)"); plt.ylabel("Density")
    plt.legend(); save("14_regression_mc")
    grid = np.linspace(x.min(),x.max(),250)
    plt.scatter(x,y,color=BLUE,alpha=.8,s=42)
    plt.plot(grid,fit["a"]+fit["b"]*grid,color=ORANGE,lw=2.5)
    plt.axvline(400,color=GREEN,ls="--",lw=1.6)
    plt.title("Hardness increases approximately linearly with temperature")
    plt.xlabel("Heat-treatment temperature (°C)"); plt.ylabel("Hardness (HV)")
    save("10_process_fit")
    plt.scatter(fit["fitted"],fit["residuals"],color=BLUE,s=42,alpha=.8)
    plt.axhline(0,color=ORANGE,ls="--")
    plt.title("Residuals reveal the spread around the fitted process line")
    plt.xlabel("Fitted hardness (HV)"); plt.ylabel("Residual hardness (HV)")
    save("11_residuals_fitted")
    stats.probplot(fit["residuals"],dist="norm",plot=plt)
    plt.title("Normal Q-Q diagnostic for regression residuals")
    plt.xlabel("Theoretical normal quantile"); plt.ylabel("Ordered residual (HV)")
    save("12_residuals_qq")
    yhat = prediction["mean_prediction"]
    mean_ci = prediction["mean_ci"]
    pred_ci = prediction["prediction_interval"]
    plt.errorbar([0,1],[yhat,yhat],
                 yerr=[[yhat-mean_ci[0],yhat-pred_ci[0]],
                       [mean_ci[1]-yhat,pred_ci[1]-yhat]],
                 fmt="o",color=BLUE,capsize=7,lw=2)
    plt.xticks([0,1],["Mean response CI","New-part PI"])
    plt.title("A new-part prediction is less precise than the process mean")
    plt.ylabel("Hardness at 400 °C (HV)")
    save("13_prediction_intervals")
    # Exclude array-valued internals from JSON; they are reproducible from data.
    public_fit = {k:v for k,v in fit.items() if k not in ["fitted","residuals"]}
    return {"fit":public_fit,"linregress_verify":library,
            "gaussian_mle_numeric_verify":numerical,"inference":infer,
            "diagnostics":diagnostics,"prediction_400":prediction,"monte_carlo":mc,
            "temperature_min":float(x.min()),"temperature_max":float(x.max())}


def main() -> dict:
    """Run both parts and export JSON + all figures."""
    sns.set_theme(style="whitegrid",context="notebook")
    rng = np.random.default_rng(SEED)
    lifetime, temperature, hardness = load_data()
    results = {"seed":SEED,"alpha":ALPHA,
               "part_a":part_a(lifetime,rng),"part_b":part_b(temperature,hardness,rng)}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(results,indent=2),encoding="utf-8")
    print(f"Saved {OUT} and {len(list(FIG.glob('*.png')))} figures")
    return results


if __name__ == "__main__":
    main()
