"""Run all five phase-one case studies and export figures and report values.

The analysis deliberately distinguishes mean comparisons (Welch inference) from
ordinal/rank comparisons (Mann--Whitney), and keeps all random draws local.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
OUT = ROOT / "reports" / "results.json"
SEED = 403109076
ALPHA = 0.05
B = 3000
PALETTE = ["#176B87", "#E29547", "#5C8A57"]


def save(name: str) -> None:
    """Save the current Matplotlib figure at publication resolution."""
    FIG.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def bootstrap_difference(a: np.ndarray, b: np.ndarray, rng: np.random.Generator,
                         statistic: str = "mean") -> tuple[float, float]:
    """Percentile 95% CI for a minus b with independent within-group resampling."""
    fn = np.mean if statistic == "mean" else np.median
    draws = np.empty(B)
    for i in range(B):
        draws[i] = fn(rng.choice(a, len(a), replace=True)) - fn(rng.choice(b, len(b), replace=True))
    return tuple(np.quantile(draws, [0.025, 0.975]))


def compare(a: np.ndarray, b: np.ndarray, rng: np.random.Generator,
            *, alternative: str = "two-sided", ordinal: bool = False) -> dict:
    """Compare a-b, reporting diagnostics, effect size and bootstrap interval.

    Welch tests the mean even when individual observations are skewed, provided
    independent groups are sufficiently large; ordinal outcomes use ranks.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert len(a) > 2 and len(b) > 2 and np.isfinite(a).all() and np.isfinite(b).all()
    sha = stats.shapiro(a).pvalue if len(a) <= 5000 else float("nan")
    shb = stats.shapiro(b).pvalue if len(b) <= 5000 else float("nan")
    lev = stats.levene(a, b, center="median").pvalue
    if ordinal:
        u = stats.mannwhitneyu(a, b, alternative=alternative)
        # U/(n1*n2) is P(A>B)+0.5*P(tie); r_rb = 2*U/(n1*n2)-1.
        effect = 2 * u.statistic / (len(a) * len(b)) - 1
        ci = bootstrap_difference(a, b, rng, "median")
        return dict(test="Mann-Whitney U", n_a=len(a), n_b=len(b), mean_a=float(a.mean()),
                    mean_b=float(b.mean()), median_a=float(np.median(a)), median_b=float(np.median(b)),
                    statistic=float(u.statistic), p=float(u.pvalue), effect=float(effect),
                    effect_name="rank-biserial r", ci=list(ci), ci_target="median difference",
                    shapiro_a=float(sha), shapiro_b=float(shb), levene=float(lev),
                    alternative=alternative)
    t = stats.ttest_ind(a, b, equal_var=False, alternative=alternative)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    se2 = va / len(a) + vb / len(b)
    df = se2**2 / ((va / len(a))**2 / (len(a)-1) + (vb / len(b))**2 / (len(b)-1))
    pooled = math.sqrt(((len(a)-1)*va + (len(b)-1)*vb) / (len(a)+len(b)-2))
    return dict(test="Welch t", n_a=len(a), n_b=len(b), mean_a=float(a.mean()),
                mean_b=float(b.mean()), sd_a=float(np.std(a, ddof=1)), sd_b=float(np.std(b, ddof=1)),
                statistic=float(t.statistic), df=float(df), p=float(t.pvalue),
                effect=float((a.mean()-b.mean())/pooled), effect_name="Cohen d",
                ci=list(bootstrap_difference(a, b, rng)), ci_target="mean difference",
                shapiro_a=float(sha), shapiro_b=float(shb), levene=float(lev),
                alternative=alternative)


def holm(ps: list[float]) -> list[float]:
    """Holm family-wise adjusted p-values, preserving original order."""
    order = np.argsort(ps)
    adjusted = np.empty(len(ps))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(ps)-rank)*ps[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def module1(rng: np.random.Generator) -> dict:
    """Sleep/lifestyle exploration and four pre-specified comparisons."""
    df = pd.read_csv(ROOT / "Sleep_health_and_lifestyle_dataset.csv", keep_default_na=False)
    required = ["Gender", "Sleep Duration", "Physical Activity Level", "Stress Level",
                "Daily Steps", "Sleep Disorder"]
    assert df[required].notna().all().all()
    for col in required[1:5]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    assert set(df["Gender"]) == {"Male", "Female"}
    assert set(df["Sleep Disorder"]) == {"None", "Insomnia", "Sleep Apnea"}
    sns.boxplot(data=df, x="Sleep Disorder", y="Stress Level", order=["None", "Insomnia", "Sleep Apnea"],
                hue="Sleep Disorder", legend=False, palette=PALETTE)
    plt.title("Stress distributions differ across sleep-disorder labels")
    plt.xlabel("Sleep-disorder label"); plt.ylabel("Stress level (1-10)")
    save("01_stress_disorder")
    sns.scatterplot(data=df, x="Sleep Duration", y="Daily Steps", hue="Stress Level", palette="viridis",
                    alpha=.72, s=38)
    plt.title("Sleep duration and daily steps show structured clusters")
    plt.xlabel("Sleep duration (hours)"); plt.ylabel("Daily steps")
    save("02_sleep_steps")
    sns.boxplot(data=df, x="Gender", y="Stress Level", hue="Gender", legend=False, palette=PALETTE[:2])
    plt.title("Stress level by gender in the supplied synthetic sample")
    plt.xlabel("Gender"); plt.ylabel("Stress level (1-10)")
    save("03_stress_gender")
    sns.scatterplot(data=df, x="Physical Activity Level", y="Daily Steps", hue="Sleep Disorder",
                    palette=PALETTE, alpha=.65, s=36)
    plt.axvline(40, color="gray", ls="--"); plt.axvline(60, color="gray", ls="--")
    plt.title("Activity thresholds separate step-count groups")
    plt.xlabel("Physical activity (minutes/day)"); plt.ylabel("Daily steps")
    save("04_activity_steps")
    female = df.loc[df.Gender == "Female", "Stress Level"].to_numpy()
    male = df.loc[df.Gender == "Male", "Stress Level"].to_numpy()
    activity = df["Physical Activity Level"]
    high = df.loc[activity >= 60, "Daily Steps"].to_numpy()
    low = df.loc[activity < 40, "Daily Steps"].to_numpy()
    no = df.loc[df["Sleep Disorder"] == "None", "Sleep Duration"].to_numpy()
    disorder = df.loc[df["Sleep Disorder"] != "None", "Sleep Duration"].to_numpy()
    men = df[df.Gender == "Male"]
    men_low = men.loc[men["Stress Level"] == 3, "Physical Activity Level"].to_numpy()
    men_high = men.loc[men["Stress Level"] >= 7, "Physical Activity Level"].to_numpy()
    # Stress is an ordinal scale. Other questions explicitly target means.
    tests = {
        "gender_stress": compare(female, male, rng, ordinal=True),
        "activity_steps": compare(high, low, rng, alternative="greater"),
        "sleep_disorder": compare(no, disorder, rng),
        "male_stress_activity": compare(men_low, men_high, rng, alternative="greater"),
    }
    adjusted = holm([v["p"] for v in tests.values()])
    for item, padj in zip(tests.values(), adjusted):
        item["holm_p"] = padj
    return {"n": len(df), "middle_activity_excluded": int(((activity >= 40)&(activity < 60)).sum()),
            "disorder_counts": df["Sleep Disorder"].value_counts().to_dict(), "tests": tests}


def module2() -> dict:
    """Binomial OC curve for accept if defectives <= 3, matching source PDF."""
    p = np.array([.01, .02, .03, .05, .10, .15, .20])
    acceptance = stats.binom.cdf(3, 60, p)
    smooth = np.linspace(0, .20, 401)
    plt.plot(100*smooth, stats.binom.cdf(3, 60, smooth), color=PALETTE[0], lw=2.5)
    plt.scatter(100*p, acceptance, color=PALETTE[1], zorder=3)
    for target in [.02, .05]:
        y = stats.binom.cdf(3, 60, target)
        plt.annotate(f"{target:.0%}: {y:.3f}", (100*target, y), xytext=(7, 10),
                     textcoords="offset points", fontsize=9)
    plt.title("Acceptance falls rapidly as defect rate rises")
    plt.xlabel("True defect rate (%)"); plt.ylabel("Probability of batch acceptance")
    plt.ylim(0, 1.04); plt.grid(alpha=.2)
    save("05_oc_curve")
    return {"n": 60, "batch_size": 2000, "acceptance_number": 3,
            "p": p.tolist(), "acceptance": acceptance.tolist(),
            "reject_at_5pct": float(1-stats.binom.cdf(3, 60, .05)),
            "accept_at_2pct": float(stats.binom.cdf(3, 60, .02))}


def module3(rng: np.random.Generator) -> dict:
    """Normal probability, simulated heights and transformation diagnostics."""
    mu, sigma, n = 182, 8, 1200
    heights = rng.normal(mu, sigma, n)
    logh = np.log(heights)
    box, lam = stats.boxcox(heights)
    fig, ax = plt.subplots()
    sns.histplot(heights, bins=30, stat="density", color=PALETTE[0], alpha=.45, ax=ax)
    x = np.linspace(heights.min(), heights.max(), 300)
    ax.plot(x, stats.norm.pdf(x, heights.mean(), heights.std(ddof=1)), color=PALETTE[1], lw=2.5)
    ax.set(title="Simulated heights match the fitted normal density", xlabel="Height (cm)", ylabel="Density")
    save("06_height_histogram")
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, values, title in zip(axs, [heights, logh, box], ["Original", "Natural log", "Box-Cox"]):
        stats.probplot(values, dist="norm", plot=ax)
        ax.set_title(title); ax.set_xlabel("Theoretical normal quantile"); ax.set_ylabel("Ordered sample")
    fig.suptitle("Transformations do not automatically improve normality", y=1.04)
    save("07_height_qq_comparison")
    for name, values, label in [("07a_height_qq_raw", heights, "Original heights"),
                                ("07b_height_qq_log", logh, "Log-transformed heights"),
                                ("07c_height_qq_boxcox", box, "Box-Cox-transformed heights")]:
        fig, ax = plt.subplots()
        stats.probplot(values, dist="norm", plot=ax)
        ax.set_title(f"Normal Q-Q diagnostic: {label}")
        ax.set_xlabel("Theoretical normal quantile")
        ax.set_ylabel("Ordered sample")
        save(name)
    # Correlation in a Q-Q plot is a descriptive shape score, not a formal test.
    qq_r2 = [float(stats.probplot(v)[1][2]**2) for v in [heights, logh, box]]
    return {"probability_175_195": float(stats.norm.cdf(195, mu, sigma)-stats.norm.cdf(175, mu, sigma)),
            "z_lower": (175-mu)/sigma, "z_upper": (195-mu)/sigma,
            "sample_mean": float(heights.mean()), "sample_sd": float(heights.std(ddof=1)),
            "boxcox_lambda": float(lam), "qq_r2": qq_r2,
            "shapiro_p": [float(stats.shapiro(v).pvalue) for v in [heights, logh, box]]}


def required_n(d: float, alpha: float = .01, target: float = .85) -> int:
    """Solve equal-size, two-sided noncentral-t power exactly by integer search."""
    def power(n: int) -> float:
        df = 2*n-2
        critical = stats.t.ppf(1-alpha/2, df)
        nc = abs(d)*math.sqrt(n/2)
        return float(stats.nct.sf(critical, df, nc) + stats.nct.cdf(-critical, df, nc))
    return next(n for n in range(2, 100000) if power(n) >= target)


def module4(rng: np.random.Generator, students: pd.DataFrame) -> dict:
    """Test-preparation mean comparison and prospective power calculation."""
    a = students.loc[students["test preparation course"] == "completed", "math score"].to_numpy()
    b = students.loc[students["test preparation course"] == "none", "math score"].to_numpy()
    result = compare(a, b, rng)
    result["alpha"] = .01
    result["required_n_per_group"] = required_n(result["effect"])
    sns.boxplot(data=students, x="test preparation course", y="math score", hue="test preparation course",
                legend=False, palette=PALETTE[:2])
    sns.stripplot(data=students, x="test preparation course", y="math score", color="#263238",
                  alpha=.17, size=2)
    plt.title("Completed preparation is associated with a higher math mean")
    plt.xlabel("Test preparation course"); plt.ylabel("Math score (0-100)")
    save("08_math_test_prep")
    ns = np.arange(20, max(320, result["required_n_per_group"]+75), 5)
    powers = []
    for n in ns:
        df = 2*n-2; crit = stats.t.ppf(1-.01/2, df)
        nc = abs(result["effect"])*math.sqrt(n/2)
        powers.append(stats.nct.sf(crit, df, nc)+stats.nct.cdf(-crit, df, nc))
    plt.plot(ns, powers, color=PALETTE[0], lw=2.5)
    plt.axhline(.85, ls="--", color=PALETTE[1]); plt.axvline(result["required_n_per_group"], ls=":", color=PALETTE[2])
    plt.title("Observed effect requires a planned equal-size sample")
    plt.xlabel("Sample size per group"); plt.ylabel("Two-sided power at alpha = 0.01")
    plt.ylim(0, 1); plt.grid(alpha=.2)
    save("09_power_curve")
    return result


def module5(rng: np.random.Generator, students: pd.DataFrame) -> dict:
    """Composite index, gender comparison and reading-writing association."""
    cols = ["math score", "reading score", "writing score"]
    z = (students[cols]-students[cols].mean())/students[cols].std(ddof=1)
    df = students.assign(composite=z.mean(axis=1))
    female = df.loc[df.gender == "female", "composite"].to_numpy()
    male = df.loc[df.gender == "male", "composite"].to_numpy()
    comparison = compare(female, male, rng)
    sns.boxplot(data=df, x="gender", y="composite", hue="gender", legend=False, palette=PALETTE[:2])
    plt.title("Composite performance differs by gender in this sample")
    plt.xlabel("Gender"); plt.ylabel("Mean of three standardized scores")
    save("10_composite_gender")
    x, y = df["reading score"].to_numpy(), df["writing score"].to_numpy()
    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    sns.regplot(x=x, y=y, scatter_kws={"s":12, "alpha":.26}, line_kws={"color":PALETTE[1]})
    plt.title(f"Reading and writing scores strongly track together (r = {pearson.statistic:.2f})")
    plt.xlabel("Reading score (0-100)"); plt.ylabel("Writing score (0-100)")
    save("11_reading_writing")
    # Pearson is primary: scatter is roughly linear; Spearman is a robust sensitivity check.
    subject_gap = (df.groupby("gender")[cols].mean().loc["female"]-
                   df.groupby("gender")[cols].mean().loc["male"]).to_dict()
    return {"n": len(df), "comparison": comparison,
            "pearson_r": float(pearson.statistic), "pearson_p": float(pearson.pvalue),
            "pearson_ci": list(pearson.confidence_interval(.95)),
            "spearman_r": float(spearman.statistic), "spearman_p": float(spearman.pvalue),
            "subject_gap_female_minus_male": subject_gap,
            "index_mean": float(df.composite.mean()), "index_sd": float(df.composite.std(ddof=1))}


def main() -> dict:
    """Run and validate all modules; export machine-readable results."""
    sns.set_theme(style="whitegrid", context="notebook")
    rng = np.random.default_rng(SEED)
    students = pd.read_csv(ROOT / "StudentsPerformance.csv")
    assert len(students) == 1000 and students.notna().all().all()
    results = {"seed": SEED, "bootstrap_resamples": B,
               "module1": module1(rng), "module2": module2(), "module3": module3(rng),
               "module4": module4(rng, students), "module5": module5(rng, students)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {OUT} and {len(list(FIG.glob('*.png')))} figures")
    return results


if __name__ == "__main__":
    main()
