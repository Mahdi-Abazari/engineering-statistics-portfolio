# Applied Statistical Inference — Phase 1

**Author:** Mahdi Abazari Aghdam

![Python 3.13](https://img.shields.io/badge/Python-3.13-blue) ![License: MIT (code)](https://img.shields.io/badge/code%20license-MIT-green) ![Reproducible](https://img.shields.io/badge/analysis-reproducible-176B87)

An English, five-case statistics portfolio built from the supplied Engineering Statistics Phase 1 assignment. It pairs a readable, **executed notebook** with reusable Python analysis, 14 high-resolution figures, machine-readable results, and a **compiled 14-page LaTeX report**. The report design takes visual cues from the supplied IT-project template, while the statistical content follows the Phase 1 assignment.

## Deliverables

- [Executed step-by-step notebook](notebooks/phase1_applied_inference.ipynb)
- [Compiled report PDF](reports/latex/main.pdf) and [editable LaTeX source](reports/latex/main.tex)
- [Results JSON](reports/results.json) and [figures](reports/figures/)
- [Reproducible analysis code](src/run_all.py)

## Key findings

| Case | Result |
|---|---|
| Sleep/lifestyle | High activity (≥60 min/day) exceeds low activity (<40) by about **2,604 daily steps** in the supplied synthetic file. The men-only low-stress comparison has **just four records** and is inconclusive after Holm adjustment. |
| Acceptance sampling | The original assignment says **reject at four or more sample defectives**. Therefore `P(accept | 2% defects) = 0.9678` and `P(reject | 5% defects) = 0.3527` under the requested binomial approximation. |
| Normal theory | `P(175 < height < 195 cm) = 0.7571` under `Normal(182, 8²)`. A simulated 1,200-player sample is already normal-looking; Box–Cox offers only a marginal in-sample Q–Q gain. |
| Test preparation | Completed-course students score **5.62 points higher** in math on average (`p = 1.04 × 10⁻⁸`, Welch). This is an association, not a causal effect. With the observed effect, **186 per group** reach approximately 85% power at two-sided α = 0.01. |
| Composite index | The female-minus-male composite mean gap is **0.252 standardized units**. Reading/writing correlate strongly (`r = 0.9546`), so an equal-weight index gives verbal performance two correlated components. |

## Reproduce from a clean environment

The pinned environment was verified with Python 3.13; a TeX installation with `pdflatex` is also required. Commands below run from **this directory**:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.run_all
python build_notebook.py
python build_report.py
cd reports\latex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

On macOS/Linux, activate with `source .venv/bin/activate` and use `cd reports/latex`.

The first three Python commands regenerate the JSON, all figures, the executed notebook, and numeric LaTeX macros. Two LaTeX passes resolve citations and cross-references. The fixed local RNG seed is `403109076`; each bootstrap interval uses 3,000 resamples. The notebooks call the same analysis functions that build the report, so the displayed output and PDF are traceable to one code path.

Run `python -m unittest discover -s tests -v` for the OC-rule, Holm-adjustment, and power-threshold checks. The included GitHub Actions workflow reruns these checks and regenerates the analysis on pushes and pull requests.

## Statistical decisions worth reviewing

- **Source conflict resolved:** the English planning brief describes the OC rule incorrectly. The original Persian assignment says reject when defectives in the sample are **≥4**; the implementation accepts **≤3**. The report documents this.
- **Estimands are explicit:** ordinal stress uses Mann–Whitney U and rank-biserial effect size; questions about group means use Welch tests. Shapiro and Levene diagnostics are reported with the results.
- **Four sleep comparisons are one family:** raw and Holm-adjusted p-values are both provided.
- **Uncertainty is visible:** every group comparison includes an effect size and 95% bootstrap interval. The four-person subgroup is identified as fragile; its bootstrap interval and Welch p-value are not used interchangeably.
- **No causal claims:** the sleep data are described as synthetic in the planning brief, and the exam data source acknowledges generated data. Test-preparation participation was not randomized.
- **OC approximation:** the assignment requests a binomial model. Exact finite-lot sampling without replacement would be hypergeometric.

## Layout

```text
phase1/
├── Sleep_health_and_lifestyle_dataset.csv
├── StudentsPerformance.csv
├── src/run_all.py
├── notebooks/phase1_applied_inference.ipynb
├── reports/
│   ├── results.json
│   ├── figures/*.png
│   └── latex/{main.tex,values.tex,main.pdf}
├── build_notebook.py
├── build_report.py
├── requirements.txt
└── LICENSE
```

The CSV files came with the course project; code's MIT license **does not grant a license for the datasets**. Dataset source pages: [Sleep Health and Lifestyle](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset) and [Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams).

## Scope

This phase is part of the [Engineering Statistics Portfolio](../README.md); [Phase 2](../phase2/README.md) covers estimation theory and regression. The report describes the supplied files and assumptions, not a clinical population or an educational intervention effect.
