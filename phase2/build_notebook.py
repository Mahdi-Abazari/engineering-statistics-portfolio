"""Build one executed, cell-by-cell notebook for the complete Phase 2 project."""
from __future__ import annotations

import base64
import contextlib
import io
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "notebooks" / "phase2_estimation_regression.ipynb"


def md(value: str):
    return nbf.v4.new_markdown_cell(value)


def code(value: str):
    return nbf.v4.new_code_cell(value)


cells = [
    md(r"""# Estimation Theory & Regression Inference | Phase 2

**Mahdi Abazari Aghdam · Sharif University of Technology**

This executed notebook answers every question in the original Phase 2 assignment. Part A estimates an exponential mean lifetime; Part B derives and tests a simple temperature-hardness regression. The reported estimators use course formulas and base arithmetic. Library fits are isolated as verification. A single fixed NumPy generator (seed `403109076`) makes all Monte Carlo and bootstrap outputs reproducible.

| Part | Steps | Main deliverable |
|---|---|---|
| A: Reliability | A1–A8 | MoM, MLE, bias, consistency, efficiency, intervals, reliability |
| B: Process regression | B1–B8 | OLS, MLE equivalence, inference, diagnostics, prediction |
"""),
    md("## 0 | Run the entire pipeline\n\nThis cell validates both supplied CSV files and regenerates the results JSON plus 14 high-resolution figures. All later cells show outputs from that single run."),
    code("""from IPython.display import display, Image
import pandas as pd
from src.run_all import main, FIG
R = main()
A, B = R['part_a'], R['part_b']
print('Seed:', R['seed'], '| Figures:', len(list(FIG.glob('*.png'))))"""),
]


def add(title: str, explanation: str, implementation: str):
    cells.append(md(f"### {title}\n\n{explanation}"))
    cells.append(code(implementation))


cells.append(md(r"# Part A | Exponential reliability estimation" + "\n\n" + r"Model: independent lifetimes $X_i \sim \mathrm{Exponential}(\theta)$ with mean and standard deviation $\theta>0$. This is an assumption to assess, not a fact established by the CSV."))
add("A1. Explore the failure-time sample",
    "Report n, mean, median, sample SD, range, and CV = sample SD / mean. The exponential model has population CV = 1 because its mean and SD both equal θ. An observed CV near 1 is supportive but cannot prove exponential fit; tail observations matter.",
    """d=A['descriptive']
display(pd.DataFrame([d]).round(3))
display(Image(filename=str(FIG/'01_lifetime_histogram.png'), width=700))""")
add("A2. Method of moments — from scratch",
    r"Equate $\bar X$ to $E[X]=\theta$. Therefore $\hat\theta_{MoM}=\bar X=\sum_iX_i/n$. The implementation in `theta_mom` uses base arithmetic; no fitting routine is in this answer path.",
    """from src.estimation import theta_mom
print('MoM mean lifetime (hours):', A['theta_mom'])""")
add("A3. Maximum likelihood — derivation and numerical check",
    r"For positive $x_i$, $\ell(\theta)=-n\log\theta-\sum_i x_i/\theta$. The score is $-n/\theta+\sum_i x_i/\theta^2$, so the stationary point is $\bar x$. The second derivative there is negative. `theta_mle` is closed form; the optimizer and `scipy.stats.expon.fit` run only in separate verification functions.",
    """print('Closed-form MLE:',A['theta_mle'])
print('Numerical MLE check:',A['theta_numeric_verify'])
print('SciPy fit check:',A['theta_scipy_verify'])""")
add("A4. Analytical bias and 10,000-trial confirmation",
    r"Linearity gives $E[\hat\theta]=E[\bar X]=\theta$ and independence gives $\mathrm{Var}(\hat\theta)=\theta^2/n$. Simulate 10,000 new samples of 60 components from Exponential($\hat\theta$). The plug-in value is a simulation parameter, not a known physical truth.",
    """display(pd.DataFrame([{'trials':A['mc_bias_trials'],'mean estimate':A['mc_estimate_mean'],
  'empirical bias':A['mc_bias'],'empirical variance':A['mc_variance'],
  'theoretical variance':A['theoretical_variance']}]).round(3))
display(Image(filename=str(FIG/'02_estimator_mc.png'), width=700))""")
add("A5. Consistency and asymptotic normality",
    r"For $n\in\{10,50,200,1000,5000\}$, 3,000 samples per size estimate variance. Theoretical variance $\theta^2/n\to0$, and unbiasedness then gives consistency in probability. At n = 5,000, standardize estimates by $\theta/\sqrt n$ and inspect a normal Q–Q plot.",
    """display(pd.DataFrame(A['consistency']).round(3))
print('Asymptotic Q-Q R²:',round(A['asymptotic_qq_r2'],6))
for name in ['03_consistency','04_consistency_loglog','05_asymptotic_qq']:
    display(Image(filename=str(FIG/f'{name}.png'),width=720))""")
add("A6. Cramér–Rao bound and efficiency",
    r"For one exponential observation, Fisher information is $I_1(\theta)=1/\theta^2$; for n independent observations, $I_n=n/\theta^2$. The unbiased-estimator bound is $\theta^2/n$. The exact variance of $\bar X$ equals that bound, so theoretical efficiency is 1. The Monte Carlo ratio can drift slightly above 1 due to finite simulation error.",
    """print('CRLB (hours²):',round(A['crlb'],3))
print('Monte Carlo variance (hours²):',round(A['mc_variance'],3))
print('CRLB / Monte Carlo variance:',round(A['efficiency'],4))
display(Image(filename=str(FIG/'06_efficiency.png'),width=670))""")
add("A7. Exact, bootstrap, and coverage-checked intervals",
    r"The pivot $2n\bar X/\theta\sim\chi^2_{2n}$ yields an exact two-sided 95% interval. A separate 5,000-resample nonparametric percentile bootstrap interval uses the observed 60 lifetimes. To assess coverage, 2,000 new 60-item datasets are generated from the plug-in exponential model; each gets an exact interval and a 1,000-resample bootstrap interval. The smaller bootstrap count in the coverage experiment is stated explicitly for computational feasibility.",
    """display(pd.DataFrame([{'method':'Exact pivot','lower':A['exact_ci'][0],
  'upper':A['exact_ci'][1],'width':A['exact_width']},
 {'method':'Percentile bootstrap','lower':A['bootstrap_ci'][0],
  'upper':A['bootstrap_ci'][1],'width':A['bootstrap_width']}]).round(3))
print('Coverage experiment:',A['coverage'])
for name in ['07_bootstrap_lifetime','08_coverage']:
    display(Image(filename=str(FIG/f'{name}.png'),width=700))""")
add("A8. Reliability at 500 hours",
    r"By MLE invariance, $\hat R(500)=\exp(-500/\hat\theta)$. Transform the 5,000 bootstrap mean estimates by the same survival function to construct a percentile interval. Reliability is a model-based probability of surviving beyond 500 hours, not an observed fraction of this sample.",
    """print('Estimated R(500):',round(A['reliability_500'],5))
print('Bootstrap 95% CI:',[round(v,5) for v in A['reliability_bootstrap_ci']])
display(Image(filename=str(FIG/'09_reliability.png'),width=700))""")

cells.append(md(r"# Part B | Temperature and hardness regression" + "\n\n" + r"Assumed model: $Y_i=A+B x_i+\varepsilon_i$ with independent Gaussian errors of common variance. Temperature is measured in °C and hardness in HV. The file order is temperature sorted, so Durbin–Watson is descriptive and cannot be treated as a time-series independence test."))
add("B1. Scatter and linearity",
    "Plot hardness against temperature and inspect whether a straight line captures the main trend. Curvature or increasing spread would weaken the simple-model assumptions.",
    """print('Temperature range (°C):',B['temperature_min'],B['temperature_max'])
display(Image(filename=str(FIG/'10_process_fit.png'),width=740))""")
add("B2. Least squares from the normal equations",
    r"Compute $S_{xx}=\sum(x_i-\bar x)^2$ and $S_{xy}=\sum(x_i-\bar x)(y_i-\bar y)$ directly; then $b=S_{xy}/S_{xx}$ and $a=\bar y-b\bar x$. `ols_fit` is the answer path. SciPy's `linregress` is used solely in `ols_linregress_verify`.",
    """f=B['fit']
display(pd.DataFrame([{'Sxx':f['sxx'],'Sxy':f['sxy'],'intercept a':f['a'],
  'slope b (HV/°C)':f['b']}]).round(5))
print('Independent library check:',B['linregress_verify'])""")
add("B3. Why Gaussian MLE equals LSE",
    r"For normal errors, $\ell(A,B,\sigma^2)=-\frac n2\log(2\pi\sigma^2)-SSE(A,B)/(2\sigma^2)$. For fixed positive variance, maximizing this log likelihood in A and B is equivalent to minimizing SSE. The score equations are the least-squares normal equations. A generic numerical optimizer fits centered intercept, scaled slope, and log sigma, then transforms coefficients back.",
    """print('Custom LSE coefficients:',f['a'],f['b'])
print('Numerical Gaussian MLE check:',B['gaussian_mle_numeric_verify'])
print('Coefficient agreement tolerance: 1e-5')""")
add("B4. Error variance and residual scale",
    r"Residuals are $e_i=y_i-a-bx_i$. The unbiased error-variance estimate divides $SSE=\sum e_i^2$ by $n-2$; residual standard error $s=\sqrt{s^2}$ is measured in HV. The Gaussian MLE of variance divides by n and is therefore numerically different, even though its coefficient MLEs equal LSE.",
    """print('SSE:',round(f['sse'],3),'HV²')
print('Unbiased s²:',round(f['s2'],3),'HV²')
print('Residual standard error s:',round(f['s'],3),'HV')
print('Gaussian MLE sigma²:',round(B['gaussian_mle_numeric_verify']['sigma2_mle'],3),'HV²')""")
add("B5. Confidence intervals and slope test",
    r"Using $t_{n-2}$, compute 95% CIs for A and B from course standard-error formulas. Test $H_0:B=0$ versus $H_1:B\ne0$ at 5%. The slope is HV per 1 °C; the intercept is extrapolated to 0 °C and has little direct process meaning. A separate 10,000-trial plug-in Gaussian simulation checks direct-sum slope bias, variance, and 95% slope-interval coverage.",
    """i=B['inference']
display(pd.DataFrame([{'parameter':'Intercept A','estimate':f['a'],'SE':i['se_a'],
 'CI lower':i['a_ci'][0],'CI upper':i['a_ci'][1]},
 {'parameter':'Slope B','estimate':f['b'],'SE':i['se_b'],
 'CI lower':i['b_ci'][0],'CI upper':i['b_ci'][1]}]).round(5))
print('Slope test: t =',round(i['slope_t'],4),'df =',i['df'],'p =',i['slope_p'])
print('Regression Monte Carlo:',B['monte_carlo'])
display(Image(filename=str(FIG/'14_regression_mc.png'),width=700))""")
add("B6. Coefficient of determination",
    r"Calculate $R^2=1-SSE/S_{yy}$ from direct sums. It describes the fraction of in-sample hardness variation captured by this line; it does not prove causation or model adequacy.",
    """print('Syy:',round(f['syy'],3),'HV²')
print('R²:',round(f['r2'],5),'= ',round(100*f['r2'],2),'% of sample variation')""")
add("B7. Residual diagnostics",
    r"Inspect residuals against fitted hardness and a normal Q–Q plot. Shapiro assesses residual normality; Breusch–Pagan regresses squared residuals on temperature and uses $nR^2$ with a chi-square(1) reference. Durbin–Watson is reported in the file's temperature order only, so it has no valid temporal interpretation.",
    """print('Diagnostics:',{k:round(v,5) for k,v in B['diagnostics'].items()})
for name in ['11_residuals_fitted','12_residuals_qq']:
    display(Image(filename=str(FIG/f'{name}.png'),width=720))""")
add("B8. Mean response and new-part prediction at 400 °C",
    r"At $x_0=400$, use $\hat y_0=a+bx_0$. A 95% confidence interval estimates the **process mean** at this temperature. A 95% prediction interval adds the variance of one new observation and is wider. For quality control of a single new part, use the prediction interval; for estimating the average process response, use the confidence interval.",
    """q=B['prediction_400']
display(pd.DataFrame([{'target':'Mean response','point':q['mean_prediction'],
  'lower':q['mean_ci'][0],'upper':q['mean_ci'][1]},
 {'target':'Single new part','point':q['mean_prediction'],
  'lower':q['prediction_interval'][0],'upper':q['prediction_interval'][1]}]).round(3))
display(Image(filename=str(FIG/'13_prediction_intervals.png'),width=700))""")

cells.append(md("""## Synthesis and reproducibility

The exponential mean is both the MoM and MLE estimate because the sample mean is the natural moment and sufficient statistic. Gaussian regression coefficient MLEs equal least squares because both optimize the same residual sum of squares. Simulation confirms theoretical bias, variance, consistency, and exact-interval coverage within finite Monte Carlo error; the bootstrap interval shows modest undercoverage in this experiment. A separate regression simulation checks coefficient bias, slope variance, and interval coverage. The two datasets are assignment-provided and realistic-looking, not independently verified observations from an operating factory.

Run `python -m src.run_all` to regenerate results and figures, `python build_notebook.py` to re-execute this notebook, and `python build_report.py` followed by two `pdflatex` passes in `reports/latex` to rebuild the PDF."""))

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.13"}})
DEST.parent.mkdir(parents=True,exist_ok=True)

# Store genuine outputs without a separate Jupyter kernel startup. All source
# cells remain ordinary executable Python when opened in Jupyter.
class ImageFile:
    def __init__(self, filename, width=None):
        self.filename, self.width = filename, width

captured = []
def capture_display(obj):
    if isinstance(obj, ImageFile):
        data = base64.b64encode(Path(obj.filename).read_bytes()).decode("ascii")
        captured.append(nbf.v4.new_output("display_data",data={"image/png":data},
                                          metadata={"image/png":{"width":obj.width}}))
    elif hasattr(obj,"to_html"):
        captured.append(nbf.v4.new_output("display_data",data={
            "text/plain":repr(obj),"text/html":obj.to_html()}))
    else:
        captured.append(nbf.v4.new_output("display_data",data={"text/plain":repr(obj)}))

scope = {"display":capture_display,"Image":ImageFile,"__name__":"__notebook__"}
counter = 0
for cell in nb.cells:
    if cell.cell_type != "code":
        continue
    counter += 1
    captured = []
    source = cell.source.replace("from IPython.display import display, Image\n", "")
    # Ordinary reruns in Jupyter import IPython display. Direct execution uses
    # capture_display so the saved notebook embeds real tables and PNG files.
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(source,f"notebook-cell-{counter}","exec"),scope)
    cell.execution_count = counter
    cell.outputs = []
    if stdout.getvalue():
        cell.outputs.append(nbf.v4.new_output("stream",name="stdout",text=stdout.getvalue()))
    cell.outputs.extend(captured)
nbf.write(nb,DEST)
print(f"Executed and saved {DEST} ({len(cells)} cells, {counter} code cells)")
