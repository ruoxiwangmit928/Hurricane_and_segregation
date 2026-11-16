#!/usr/bin/env python3
"""Income associations with environmental adjustment in the original Top-100.

The cohort is the original Top-100 roster. Left joins retain its members before
an explicit common complete-case screen. The model sample size and exclusions
are calculated from the available observations and recorded in audit tables.

Inputs are CSV files with unique string cbg_id keys:
  --roster: exactly 100 original CBG IDs; other columns are ignored.
  --outcomes: median_income_current_figure3, peak_net_population_increase,
      current_host_mixing_decline_percent_baseline_S,
      current_host_exploration_reduction.
  --covariates: log_distance_coastal_water, flood100_share,
      flood500_only_share, log_damage_per_household,
      distance_to_coastal_water_centroid_km, damage_per_household.

Mixing is 100*(S_post-S_pre)/S_pre = 100*(M_pre-M_post)/(1-M_pre),
not percent change relative to baseline M. Pre/post are weeks 0-5 and 8-11.
Exploration is the reduction in distinct visited locations / total visits.
The population outcome is peak net population increase, not distinct arrivals.
The source column damage_per_household is a supplementary FEMA monetary
indicator, not a verified measure of total physical losses.

Fits use OLS with HC3 covariance and normal-reference confidence intervals.
All three outcomes and four nested models use the same complete-case sample.
Tables and sample/exclusion audits are always saved. --plots also generates
the two corresponding figures. Requires numpy, pandas, scipy, statsmodels;
optional plots require matplotlib. No raw device data or notebook is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats
import statsmodels
import statsmodels.api as sm


INCOME = "median_income_current_figure3"
OUTCOMES = [
    "peak_net_population_increase",
    "current_host_mixing_decline_percent_baseline_S",
    "current_host_exploration_reduction",
]
CONTROLS = [
    "log_distance_coastal_water", "flood100_share", "flood500_only_share",
    "log_damage_per_household",
]
RAW = ["distance_to_coastal_water_centroid_km", "damage_per_household"]
SPECIFICATIONS = {
    "income_only": [], "plus_distance": CONTROLS[:1],
    "plus_flood": CONTROLS[:3], "plus_fema": CONTROLS,
}


def read_keyed(path: Path, required: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"cbg_id": str})
    missing = set(["cbg_id", *required]).difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    frame["cbg_id"] = frame.cbg_id.str.strip()
    if frame.cbg_id.isna().any() or frame.cbg_id.eq("").any() or frame.cbg_id.duplicated().any():
        raise ValueError(f"{path.name}: cbg_id must be nonmissing and unique")
    frame = frame[["cbg_id", *required]].copy()
    for column in required:
        # Nonempty malformed values fail; ordinary missing values are audited.
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def prepare_sample(roster_path: Path, outcomes_path: Path, covariates_path: Path):
    roster = read_keyed(roster_path, [])
    if len(roster) != 100:
        raise ValueError(f"The original Top-100 roster must contain 100 unique CBGs; found {len(roster)}")
    outcomes = read_keyed(outcomes_path, [INCOME, *OUTCOMES])
    covariates = read_keyed(covariates_path, [*CONTROLS, *RAW])
    for name, table in [("outcomes", outcomes), ("covariates", covariates)]:
        extra = set(table.cbg_id) - set(roster.cbg_id)
        if extra:
            raise ValueError(f"{name}: contains {len(extra)} CBGs outside the original roster")
    merged = roster.merge(outcomes, on="cbg_id", how="left", validate="one_to_one", indicator="outcome_join")
    merged = merged.merge(covariates, on="cbg_id", how="left", validate="one_to_one", indicator="covariate_join")
    needed = [INCOME, *OUTCOMES, *CONTROLS, *RAW]
    finite = np.isfinite(merged[needed].to_numpy(dtype=float))
    audit = merged[["cbg_id", "outcome_join", "covariate_join"]].copy()
    audit["included"] = finite.all(axis=1)
    audit["missing_or_nonfinite_fields"] = [
        ";".join(column for column, ok in zip(needed, row) if not ok)
        for row in finite
    ]
    audit["exclusion_reason"] = [
        ";".join([
            *(["no_outcome_record"] if row.outcome_join != "both" else []),
            *(["no_covariate_record"] if row.covariate_join != "both" else []),
            *(["missing_or_nonfinite_required_fields"] if not row.included else []),
        ]) for row in audit.itertuples()
    ]
    data = merged.loc[audit.included, ["cbg_id", *needed]].copy()
    if data.empty:
        raise ValueError("No common complete observations remain")
    if not data[INCOME].gt(0).all():
        raise ValueError("Income must be positive in the complete sample")
    for logged, raw in [(CONTROLS[0], RAW[0]), (CONTROLS[-1], RAW[-1])]:
        if not data[raw].ge(0).all():
            raise ValueError(f"{raw} must be nonnegative")
        np.testing.assert_allclose(data[logged], np.log1p(data[raw]), atol=1e-12)
    if not data[["flood100_share", "flood500_only_share"]].ge(0).all().all() or not data[["flood100_share", "flood500_only_share"]].le(1).all().all():
        raise ValueError("Flood shares must lie in [0, 1]")
    data["income_10k_current"] = data[INCOME] / 10000
    flow = pd.DataFrame([
        {"stage": "original_roster", "n_cbgs": len(roster)},
        {"stage": "with_outcome_record", "n_cbgs": int(audit.outcome_join.eq("both").sum())},
        {"stage": "with_covariate_record", "n_cbgs": int(audit.covariate_join.eq("both").sum())},
        {"stage": "with_both_records", "n_cbgs": int((audit.outcome_join.eq("both") & audit.covariate_join.eq("both")).sum())},
        {"stage": "common_complete_analysis_sample", "n_cbgs": len(data)},
        {"stage": "excluded_from_analysis", "n_cbgs": len(roster) - len(data)},
    ])
    return data, audit, flow


def fit_models(data: pd.DataFrame):
    fits, estimates, changes, predictions = {}, [], [], []
    grid = np.linspace(data[INCOME].min(), data[INCOME].max(), 201)
    for outcome in OUTCOMES:
        for specification, controls in SPECIFICATIONS.items():
            columns = ["income_10k_current", *controls]
            design = sm.add_constant(data[columns], has_constant="add")
            if np.linalg.matrix_rank(design) != design.shape[1] or len(data) <= design.shape[1]:
                raise ValueError(f"{specification}: insufficient sample or rank-deficient design")
            model = sm.OLS(data[outcome], design).fit(cov_type="HC3", use_t=False)
            if not np.isfinite(model.params).all() or not np.isfinite(model.bse).all():
                raise ValueError(f"{outcome}/{specification}: nonfinite model result")
            fits[outcome, specification] = model
            lo, hi = model.conf_int().loc["income_10k_current"]
            estimates.append({
                "outcome": outcome, "specification": specification, "n": int(model.nobs),
                "income_coefficient_per_10000": model.params.income_10k_current,
                "ci_low": lo, "ci_high": hi, "p_value": model.pvalues.income_10k_current,
                "controls": ";".join(controls), "inference": "HC3; normal approximation",
            })
            if specification in ["income_only", "plus_fema"]:
                xp = pd.DataFrame({"income_10k_current": grid / 10000})
                for column in controls:
                    xp[column] = data[column].mean()
                xp = sm.add_constant(xp, has_constant="add")[model.model.exog_names]
                predictions.append(model.get_prediction(xp).summary_frame(alpha=.05).assign(
                    outcome=outcome, specification=specification, median_income=grid))
        original = fits[outcome, "income_only"].params.income_10k_current
        adjusted = fits[outcome, "plus_fema"].params.income_10k_current
        changes.append({
            "outcome": outcome, "income_only_coefficient": original,
            "adjusted_coefficient": adjusted,
            "relative_coefficient_change_pct": 100 * (adjusted / original - 1) if original != 0 else np.nan,
        })
    risk = []
    for variable in [RAW[0], "flood100_share", RAW[1]]:
        rho, pvalue = stats.spearmanr(data[variable], data[INCOME])
        risk.append({"variable": variable, "spearman_rho": rho, "p_value": pvalue, "n": len(data)})
    return pd.DataFrame(estimates), pd.DataFrame(changes), pd.DataFrame(risk), pd.concat(predictions, ignore_index=True)


def plot_figures(data, predictions, risk, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42})
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained", sharey=True)
    for ax, variable, title, xlabel in zip(axes, [RAW[0], "flood100_share", RAW[1]],
            ["Coast/bay distance", "1% annual-chance flood zone", "FEMA monetary indicator"],
            ["Distance (km)", "Share of CBG land area", "FEMA indicator per household"]):
        x, y = data[variable].to_numpy(), data[INCOME].to_numpy()
        grid = np.linspace(x.min(), x.max(), 100)
        rng = np.random.default_rng(307)
        bootstrap = []
        for _ in range(2000):
            indices = rng.integers(0, len(x), len(x))
            bootstrap.append(np.polyval(np.polyfit(x[indices], y[indices], 1), grid))
        lo, hi = np.percentile(bootstrap, [2.5, 97.5], axis=0)
        ax.fill_between(grid, lo, hi, color=".7", alpha=.5, linewidth=0)
        ax.scatter(x, y, s=32, color="#3F8FBD", alpha=.62)
        ax.plot(grid, np.polyval(np.polyfit(x, y, 1), grid), color="black", lw=1.8)
        record = risk[risk.variable.eq(variable)].iloc[0]
        ax.set_title(f"{title}\n$r_s$ = {record.spearman_rho:.2f}, $p$ = {record.p_value:.3f}")
        ax.set(xlabel=xlabel, ylabel="Median household income ($)")
    for extension in ["png", "pdf"]:
        figure.savefig(output / f"income_environment_correlations.{extension}", dpi=300, bbox_inches="tight")
    plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.7), layout="constrained")
    labels = ["Peak net population increase", "Normalized mixing decline (%)", "Reduction in distinct locations / visits"]
    for ax, outcome, ylabel in zip(axes, OUTCOMES, labels):
        selected = predictions[predictions.outcome.eq(outcome)]
        adjusted = selected[selected.specification.eq("plus_fema")]
        ax.fill_between(adjusted.median_income.to_numpy(), adjusted.mean_ci_lower.to_numpy(), adjusted.mean_ci_upper.to_numpy(), color="#356F93", alpha=.15)
        ax.scatter(data[INCOME], data[outcome], s=30, color="#6C9EB9", alpha=.65)
        for specification, color, linestyle, label in [
            ("plus_fema", "#245C80", "-", "Environment + FEMA adjusted"),
            ("income_only", "#E66B13", "--", "Income only"),
        ]:
            values = selected[selected.specification.eq(specification)]
            ax.plot(values.median_income, values["mean"], color=color, linestyle=linestyle, label=label)
        ax.set(xlabel="Median household income ($)", ylabel=ylabel)
        ax.legend(frameon=False, fontsize=8)
    for extension in ["png", "pdf"]:
        figure.savefig(output / f"income_environment_adjustment.{extension}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def run(roster: Path, outcomes: Path, covariates: Path, output: Path, plots: bool = False):
    data, inclusion, flow = prepare_sample(roster, outcomes, covariates)
    models, changes, risk, predictions = fit_models(data)
    output.mkdir(parents=True, exist_ok=True)
    for filename, table in [
        ("analysis_inputs.csv", data), ("sample_inclusion.csv", inclusion),
        ("excluded_cbgs.csv", inclusion[~inclusion.included]), ("sample_flow.csv", flow),
        ("income_adjustment_models.csv", models), ("coefficient_changes.csv", changes),
        ("income_risk_correlations.csv", risk), ("adjusted_predictions.csv", predictions),
    ]:
        table.to_csv(output / filename, index=False)
    audit = {
        "cohort_size": int(flow.loc[flow.stage.eq("original_roster"), "n_cbgs"].iloc[0]),
        "analysis_n": len(data), "excluded_n": int((~inclusion.included).sum()),
        "selection": "Left join original 100-CBG roster, then common complete-case screen; no manual exclusions",
        "input_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in [
            ("roster", roster), ("outcomes", outcomes), ("covariates", covariates)]},
        "mixing_normalization": "100*(S_post-S_pre)/S_pre = 100*(M_pre-M_post)/(1-M_pre)",
        "pre_weeks": [0, 5], "post_weeks": [8, 11],
        "inference": "OLS HC3; normal-reference 95% intervals and two-sided p values",
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "statsmodels": statsmodels.__version__},
    }
    (output / "analysis_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if plots:
        plot_figures(data, predictions, risk, output)
    print(f"Original cohort: {audit['cohort_size']} CBGs; complete analysis: {len(data)}; excluded: {audit['excluded_n']}")
    print(f"Saved results and explicit sample/exclusion audits to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--covariates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plots", action="store_true", help="Also render the two figures")
    args = parser.parse_args()
    run(args.roster, args.outcomes, args.covariates, args.output, args.plots)
