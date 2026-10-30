"""Simple linear regression from normal equations, with separate verification.

Reported fit and interval calculations use direct sums. SciPy fitting appears
only in *_verify; optimization checks the Gaussian MLE independently.
"""
from __future__ import annotations

import math
import numpy as np
from scipy import optimize, stats


def ols_fit(x: np.ndarray, y: np.ndarray) -> dict:
    """LSE: b=Sxy/Sxx and a=ybar-b*xbar, using direct arithmetic only."""
    xx, yy = [float(v) for v in x], [float(v) for v in y]
    n = len(xx)
    if n < 3 or len(yy) != n or any(not math.isfinite(v) for v in xx+yy):
        raise ValueError("At least three finite paired observations are required")
    xbar, ybar = sum(xx)/n, sum(yy)/n
    sxx = sum((v-xbar)**2 for v in xx)
    sxy = sum((xx[i]-xbar)*(yy[i]-ybar) for i in range(n))
    syy = sum((v-ybar)**2 for v in yy)
    if sxx <= 0 or syy <= 0:
        raise ValueError("Both variables must vary")
    b = sxy/sxx
    a = ybar-b*xbar
    fitted = [a+b*v for v in xx]
    residuals = [yy[i]-fitted[i] for i in range(n)]
    sse = sum(v*v for v in residuals)
    return {"n":n, "a":a, "b":b, "xbar":xbar, "ybar":ybar,
            "sxx":sxx, "sxy":sxy, "syy":syy, "sse":sse,
            "s2":sse/(n-2), "s":math.sqrt(sse/(n-2)),
            "r2":1-sse/syy, "fitted":fitted, "residuals":residuals}


def ols_linregress_verify(x: np.ndarray, y: np.ndarray) -> dict:
    """Independent library fit used only to verify the from-scratch answer."""
    result = stats.linregress(x, y)
    return {"a":float(result.intercept), "b":float(result.slope),
            "r2":float(result.rvalue**2)}


def regression_mle_numeric_verify(x: np.ndarray, y: np.ndarray) -> dict:
    """Minimize Gaussian NLL over centered intercept, scaled slope, log sigma.

    x is centered and divided by 100 only for optimizer conditioning. The
    returned coefficients are transformed back to the original temperature.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    xbar = float(x.mean())
    z = (x-xbar)/100
    n = len(x)
    def objective(params):
        intercept_centered, slope_scaled, log_sigma = params
        sigma2 = math.exp(2*log_sigma)
        residuals = y-intercept_centered-slope_scaled*z
        sse = float(residuals@residuals)
        value = n/2*math.log(2*math.pi)+n*log_sigma+sse/(2*sigma2)
        gradient = np.array([-residuals.sum()/sigma2,
                             -(z@residuals)/sigma2,
                             n-sse/sigma2])
        return value, gradient
    start = [float(y.mean()), 0.0, math.log(float(y.std(ddof=1)))]
    result = optimize.minimize(objective, start, jac=True, method="BFGS", tol=1e-10)
    b = float(result.x[1]/100)
    a = float(result.x[0]-b*xbar)
    return {"a":a, "b":b, "sigma2_mle":float(math.exp(2*result.x[2])),
            "optimizer_success":bool(result.success),
            "gradient_norm":float(np.linalg.norm(result.jac))}


def coefficient_inference(fit: dict, alpha: float = .05) -> dict:
    """Two-sided t intervals and H0:B=0 test from course formulas."""
    n, s, sxx, xbar, a, b = [fit[k] for k in ["n","s","sxx","xbar","a","b"]]
    df = n-2
    critical = float(stats.t.ppf(1-alpha/2, df))
    se_b = s/math.sqrt(sxx)
    se_a = s*math.sqrt(1/n+xbar*xbar/sxx)
    t = b/se_b
    return {"df":df, "t_critical":critical, "se_a":se_a, "se_b":se_b,
            "a_ci":[a-critical*se_a, a+critical*se_a],
            "b_ci":[b-critical*se_b, b+critical*se_b],
            "slope_t":t, "slope_p":float(2*stats.t.sf(abs(t),df))}


def response_intervals(fit: dict, x0: float = 400, alpha: float = .05) -> dict:
    """Mean-response CI and new-part PI at x0 using t(n-2)."""
    n, xbar, sxx, s, a, b = [fit[k] for k in ["n","xbar","sxx","s","a","b"]]
    y0 = a+b*x0
    leverage = 1/n+(x0-xbar)**2/sxx
    critical = float(stats.t.ppf(1-alpha/2,n-2))
    mean_half = critical*s*math.sqrt(leverage)
    prediction_half = critical*s*math.sqrt(1+leverage)
    return {"x0":x0, "mean_prediction":y0, "leverage":leverage,
            "mean_ci":[y0-mean_half,y0+mean_half],
            "prediction_interval":[y0-prediction_half,y0+prediction_half]}


def residual_diagnostics(x: np.ndarray, fit: dict) -> dict:
    """Shapiro, Breusch-Pagan LM, and Durbin-Watson in supplied row order.

    BP regresses squared residuals on an intercept and x, with LM=n*R² and
    chi-square(1) reference. DW has no temporal interpretation if row order is
    merely temperature order; it is reported descriptively only.
    """
    residuals = np.asarray(fit["residuals"],float)
    bp_fit = ols_fit(np.asarray(x,float), residuals**2)
    lm = fit["n"]*bp_fit["r2"]
    dw = float(np.diff(residuals)@np.diff(residuals)/(residuals@residuals))
    return {"shapiro_w":float(stats.shapiro(residuals).statistic),
            "shapiro_p":float(stats.shapiro(residuals).pvalue),
            "bp_lm":lm, "bp_p":float(stats.chi2.sf(lm,1)),
            "durbin_watson":dw}


def regression_mc(x: np.ndarray, fit: dict, trials: int,
                  rng: np.random.Generator) -> tuple[dict, np.ndarray]:
    """Confirm OLS bias/variance and slope-CI coverage under plug-in normal errors.

    The direct centered-sum formulas are used on every simulated response.
    This is a model check, not evidence that factory errors are truly Gaussian.
    """
    x = np.asarray(x,float)
    n = len(x)
    x_centered = x-fit["xbar"]
    simulated_y = fit["a"]+fit["b"]*x[None,:]+rng.normal(0,fit["s"],size=(trials,n))
    ybar = simulated_y.mean(axis=1)
    slopes = ((simulated_y-ybar[:,None])@x_centered)/fit["sxx"]
    intercepts = ybar-slopes*fit["xbar"]
    residuals = simulated_y-intercepts[:,None]-slopes[:,None]*x[None,:]
    s2 = np.sum(residuals**2,axis=1)/(n-2)
    critical = stats.t.ppf(.975,n-2)
    halfwidth = critical*np.sqrt(s2/fit["sxx"])
    covered = np.mean((slopes-halfwidth <= fit["b"]) & (fit["b"] <= slopes+halfwidth))
    summary = {"trials":trials,"intercept_bias":float(intercepts.mean()-fit["a"]),
               "slope_bias":float(slopes.mean()-fit["b"]),
               "intercept_variance_empirical":float(intercepts.var(ddof=1)),
               "intercept_variance_theoretical":fit["s2"]*(1/n+fit["xbar"]**2/fit["sxx"]),
               "slope_variance_empirical":float(slopes.var(ddof=1)),
               "slope_variance_theoretical":fit["s2"]/fit["sxx"],
               "slope_ci_coverage":float(covered)}
    return summary,slopes
