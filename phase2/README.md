# Estimation Theory & Regression Inference — Phase 2

**Author:** Mahdi Abazari Aghdam · Sharif University of Technology

![Python 3.13](https://img.shields.io/badge/Python-3.13-blue) ![Code license: MIT](https://img.shields.io/badge/code%20license-MIT-green) ![Reproducible](https://img.shields.io/badge/analysis-reproducible-176B87)

An English, from-scratch statistical-inference portfolio on **component reliability** and **heat-treatment process regression**. It answers all 16 tasks (A1–A8 and B1–B8) in the supplied Phase 2 assignment. The project includes one executed, step-by-step Jupyter notebook, a formal compiled report with the Sharif logo and author's name, 14 high-resolution figures, machine-readable results, and independent verification of custom estimators.

## Deliverables

- [Executed notebook](notebooks/phase2_estimation_regression.ipynb) — one cell-by-cell explanation of A1–A8 and B1–B8
- [Compiled report PDF](reports/latex/main.pdf) and [editable source](reports/latex/main.tex)
- [Results JSON](reports/results.json) and [all figures](reports/figures/)
- [Direct-sum estimation](src/estimation.py), [direct-sum regression](src/regression.py), and [one-command pipeline](src/run_all.py)

## Key results

| Part | Result |
|---|---|
| Exponential lifetime | MoM = MLE = **356.95 hours** for 60 components. Sample CV = **1.204**; an exponential model is plausible but not established. |
| Bias and efficiency | Across **10,000** new samples, empirical bias is **−0.267 hours**. Empirical estimator variance is **2,084.36 hours²**, compared with theoretical variance and CRLB **2,123.56 hours²**. The empirical CRLB/variance ratio is **1.019**, a finite-simulation fluctuation around theoretical efficiency 1. |
| Confidence intervals | Exact 95% CI for mean lifetime: **[281.41, 467.76] hours**. Percentile bootstrap (5,000 resamples): **[261.17, 476.96] hours**. In 2,000 model-generated datasets, exact and bootstrap coverage were **95.1%** and **93.75%** respectively (1,000 bootstrap resamples per repeated dataset). |
| Reliability | Estimated survival probability beyond 500 hours: **0.2464**; bootstrap 95% CI **[0.1474, 0.3505]**. |
| Regression | Direct-sum fit: **hardness = 108.602 + 0.73332 × temperature** (HV and °C), with **R² = 0.9338**. The slope test gives **p = 5.24 × 10⁻²⁴**. |
| Regression simulation | Across **10,000** plug-in Gaussian datasets, direct-sum slope bias is **−0.00043 HV/°C** and empirical 95% slope-interval coverage is **95.25%**. |
| Prediction at 400 °C | Mean predicted hardness **401.93 HV**; mean-response 95% CI **[393.78, 410.08] HV**; new-part 95% prediction interval **[353.89, 449.97] HV**. |

## Reproduce

The pinned environment was verified with Python 3.13. Run from **this directory**:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.run_all
python build_notebook.py
python build_report.py
cd reports\latex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

On macOS/Linux, activate with `source .venv/bin/activate` and use `cd reports/latex`. The two TeX passes resolve the clickable contents, citations, and figure references. The fixed seed is `403109076`; all Monte Carlo trial counts and bootstrap counts are reported in the notebook and PDF. `python -m src.run_all` regenerates every number and figure; `build_report.py` generates numerical LaTeX macros from the JSON, avoiding hand-copied result values.

## Statistical design

- **Reported-answer path:** `theta_mom`, `theta_mle`, and `ols_fit` use direct arithmetic and course formulas. `*_verify` functions isolate SciPy fitting and generic numerical likelihood optimization.
- **Three validation layers:** mathematical derivations in the report, numerical estimates on the supplied files, and Monte Carlo confirmation of bias, variance, consistency, efficiency, and interval coverage.
- **Exact versus approximate uncertainty:** the chi-square interval is exact *under* an independent exponential model; the percentile bootstrap interval is approximate and showed modest undercoverage in the plug-in simulation.
- **Regression diagnostics:** residual plots, normal Q–Q, Shapiro–Wilk, and Breusch–Pagan support assumption assessment. Durbin–Watson is reported only descriptively: the rows are temperature sorted, not a time series.
- **Prediction targets:** the mean-response interval addresses average hardness at 400 °C; the wider prediction interval addresses one new part.
- **Scope:** the supplied CSV files are realistic-looking course data. The results do not establish a verified factory failure mechanism or a causal effect of heat-treatment temperature.

## Project structure

```text
phase2/
├── lifetimes.csv
├── process.csv
├── src/
│   ├── estimation.py
│   ├── regression.py
│   └── run_all.py
├── notebooks/phase2_estimation_regression.ipynb
├── reports/
│   ├── results.json
│   ├── figures/*.png
│   └── latex/{main.tex,values.tex,main.pdf,assets/sharif_logo.png}
├── tests/test_engine.py
├── build_notebook.py
├── build_report.py
├── requirements.txt
└── LICENSE
```

The MIT license applies to the original code and documentation; it does **not** grant rights to the assignment-provided CSV files or course brief. This phase is part of the [Engineering Statistics Portfolio](../README.md), alongside [Phase 1](../phase1/README.md).
