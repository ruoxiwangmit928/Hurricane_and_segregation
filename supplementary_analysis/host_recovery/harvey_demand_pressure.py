"""Harvey demand-pressure analysis from the saved balanced POI-week panel.

Example:
    python harvey_demand_pressure.py --panel poi_week_pressure_panel.csv.gz --output results

The input is the existing status-verified, baseline-selected, 12-week panel.
This program does not rebuild visits or infer displacement/return dates.
The pressure proxy is the stored expected-host + evacuee + other-user demand
relative to the baseline all-user 90th percentile. It is not venue capacity.

Preserves the original two-way POI/week absorption, POI-cluster covariance,
absorbed-effect degrees-of-freedom correction and cluster-t inference.
Positive-log outcomes use their own observed positive support; no zero is
invented or silently converted to a positive observation. Only aggregate
summaries, coefficients and figures are written, never POI/person records.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster

BASELINE_WEEKS = list(range(6))
ACUTE_WEEKS = [7, 8]
RECOVERY_WEEKS = [9, 10, 11]

class AbsorbedFEResult:
    def __init__(self, params, covariance, inference_df):
        self.params = params
        self.covariance = pd.DataFrame(
            covariance, index=params.index, columns=params.index
        )
        self.inference_df = inference_df
        self.bse = pd.Series(
            np.sqrt(np.clip(np.diag(covariance), 0, None)), index=params.index
        )
        self.tvalues = self.params / self.bse
        self.pvalues = pd.Series(
            2 * scipy_stats.t.sf(np.abs(self.tvalues), df=inference_df),
            index=params.index,
        )

    def conf_int(self, alpha=.05):
        critical = scipy_stats.t.ppf(1 - alpha / 2, df=self.inference_df)
        return pd.DataFrame({
            0: self.params - critical * self.bse,
            1: self.params + critical * self.bse,
        })

    def wald_test(self, restriction):
        restriction = np.atleast_2d(np.asarray(restriction, dtype=float))
        estimate = restriction @ self.params.to_numpy()
        variance = restriction @ self.covariance.to_numpy() @ restriction.T
        q = int(np.linalg.matrix_rank(restriction))
        statistic = float(estimate.T @ np.linalg.pinv(variance) @ estimate / q)
        p_value = float(scipy_stats.f.sf(statistic, q, self.inference_df))
        return SimpleNamespace(
            statistic=statistic,
            pvalue=p_value,
            df_num=q,
            df_denom=self.inference_df,
        )


def bipartite_fe_components(data, first_fe, second_fe):
    # Connected components in the two-way-FE incidence graph.
    parent = {}

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    edges = data[[first_fe, second_fe]].drop_duplicates()
    for left, right in edges.itertuples(index=False, name=None):
        union((0, left), (1, right))
    return len({find(node) for node in parent})


fe_structure_cache = {}


def two_way_fe_structure(data, fe_cols):
    hashed = pd.util.hash_pandas_object(data[list(fe_cols)], index=True).values
    key = (tuple(fe_cols), len(data), hashlib.sha256(hashed.tobytes()).hexdigest())
    if key in fe_structure_cache:
        return fe_structure_cache[key]

    n_levels = [data[fe].nunique() for fe in fe_cols]
    is_complete_grid = (
        len(data) == n_levels[0] * n_levels[1]
        and not data.duplicated(list(fe_cols)).any()
    )
    n_components = 1 if is_complete_grid else bipartite_fe_components(
        data, fe_cols[0], fe_cols[1]
    )
    absorbed_df = sum(n_levels) - n_components
    fe_structure_cache[key] = (n_components, absorbed_df)
    return n_components, absorbed_df


def fit_absorbed_fe(data, outcome, regressors, fe_cols=("place_id", "week"),
                    cluster_col="place_id", tol=1e-10, max_iter=500):
    if len(fe_cols) != 2:
        raise ValueError("This implementation expects exactly two fixed-effect dimensions")
    needed = list(dict.fromkeys([outcome] + regressors + list(fe_cols) + [cluster_col]))
    n_input_rows = len(data)
    d = data[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if d.empty:
        raise ValueError("No complete observations for model")

    model_cols = [outcome] + regressors
    z = d[model_cols].astype(float).copy()
    z -= z.mean()

    converged = False
    for iteration in range(1, max_iter + 1):
        previous = z.to_numpy(copy=True)
        for fe in fe_cols:
            z -= z.groupby(d[fe], observed=True).transform("mean")
        change = float(np.nanmax(np.abs(z.to_numpy() - previous)))
        if change < tol:
            converged = True
            break
    if not converged:
        raise RuntimeError(f"Fixed-effect absorption did not converge after {max_iter} iterations")

    X = z[regressors].loc[:, z[regressors].std() > 1e-12]
    dropped = sorted(set(regressors) - set(X.columns))
    if dropped:
        raise ValueError(f"Regressors absorbed or invariant after FE removal: {dropped}")
    if np.linalg.matrix_rank(X.to_numpy()) < X.shape[1]:
        raise ValueError("Regressor matrix is rank deficient after fixed-effect absorption")

    base_fit = sm.OLS(z.loc[X.index, outcome], X).fit()
    groups = d.loc[X.index, cluster_col]
    n_clusters = groups.nunique()
    if n_clusters < 2:
        raise ValueError("Cluster-robust inference requires at least two POI clusters")

    n_components, absorbed_df = two_way_fe_structure(d.loc[X.index], fe_cols)
    residual_df = len(d) - len(X.columns) - absorbed_df
    if residual_df <= 0:
        raise ValueError(
            f"Non-positive residual degrees of freedom after absorbed effects: {residual_df}"
        )
    cr0 = cov_cluster(base_fit, groups, use_correction=False)
    finite_sample_factor = (
        n_clusters / (n_clusters - 1)
        * (len(d) - 1) / residual_df
    )
    covariance = cr0 * finite_sample_factor
    fit = AbsorbedFEResult(base_fit.params, covariance, n_clusters - 1)

    return fit, {
        "n_input_rows": n_input_rows,
        "n_obs": len(d),
        "n_dropped_incomplete": n_input_rows - len(d),
        "n_pois": d.place_id.nunique(),
        "n_clusters": n_clusters,
        "fe_connected_components": n_components,
        "absorbed_fe_df": absorbed_df,
        "residual_df_after_absorbed_fe": residual_df,
        "cluster_inference_df": n_clusters - 1,
        "cluster_finite_sample_factor": finite_sample_factor,
        "zero_outcome_share_in_estimation_sample": float(d[outcome].eq(0).mean()),
        "iterations": iteration,
        "converged": converged,
    }


def load_panel(path):
    columns = ["place_id", "week", "host_visits", "host_total_dwell", "all_visits",
               "unique_evacuees", "capacity_normalized_pressure"]
    panel = pd.read_csv(path, usecols=columns, dtype={"place_id": "string"})
    if panel[columns].isna().any().any() or panel.duplicated(["place_id", "week"]).any():
        raise ValueError("Missing required values or duplicate POI-week records")
    numeric = columns[1:]
    if not np.isfinite(panel[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite required numeric input")
    if (panel[numeric] < 0).any().any():
        raise ValueError("Negative week, count, dwell or demand pressure")
    if set(panel.week) != set(range(12)) or not panel.groupby("place_id").week.nunique().eq(12).all():
        raise ValueError("Expected the original complete 12-week balanced POI panel")
    if not panel.groupby("place_id").unique_evacuees.nunique().eq(1).all():
        raise ValueError("Evacuee exposure must be fixed within POI")
    for column in ["week", "host_visits", "all_visits", "unique_evacuees"]:
        if not panel[column].eq(panel[column].round()).all():
            raise ValueError(f"Expected integer-valued {column}")
    panel["shared_ge3"] = panel.unique_evacuees.ge(3)
    panel["shared_any"] = panel.unique_evacuees.ge(1)
    panel["host_active"] = panel.host_visits.gt(0)
    panel["host_mean_dwell_per_visit"] = panel.host_total_dwell / panel.host_visits.replace(0, np.nan)
    panel["log1p_unique_evacuees"] = np.log1p(panel.unique_evacuees)
    panel["log1p_host_visits"] = np.log1p(panel.host_visits)
    panel["log1p_host_total_dwell"] = np.log1p(panel.host_total_dwell)
    panel["any_host_activity"] = panel.host_active.astype(float)
    panel["log_host_visits_positive"] = np.log(panel.host_visits.where(panel.host_visits.gt(0)))
    panel["log_host_total_dwell_positive"] = np.log(panel.host_total_dwell.where(panel.host_total_dwell.gt(0)))
    panel["log_mean_host_dwell_per_visit"] = panel.log_host_total_dwell_positive - panel.log_host_visits_positive
    panel["log1p_mean_host_dwell_per_visit"] = np.log1p(panel.host_mean_dwell_per_visit)
    return panel


def coefficient_row(fit, support, outcome, specification, term):
    ci = fit.conf_int().loc[term]
    return dict(outcome=outcome, specification=specification, term=term,
                coefficient=fit.params[term], std_error=fit.bse[term],
                p_value=fit.pvalues[term], ci_lower=ci.iloc[0], ci_upper=ci.iloc[1], **support)


def estimate_models(panel, pretrend=False):
    recovery = panel.loc[panel.week.isin(BASELINE_WEEKS + RECOVERY_WEEKS)].copy()
    recovery["exposure_recovery"] = recovery.log1p_unique_evacuees * recovery.week.isin(RECOVERY_WEEKS).astype(int)
    specs = [
        ("log1p_host_visits", "balanced_count"),
        ("log1p_host_total_dwell", "balanced_total_dwell"),
        ("any_host_activity", "balanced_activity_probability_lpm"),
        ("log_host_visits_positive", "conditional_positive_log_visits"),
        ("log_host_total_dwell_positive", "conditional_positive_log_total_dwell"),
        ("log_mean_host_dwell_per_visit", "conditional_log_mean_dwell_per_visit"),
        ("log1p_mean_host_dwell_per_visit", "conditional_log1p_mean_dwell_per_visit"),
    ]
    pooled_rows = []
    for outcome, label in specs:
        for controls, suffix in [([], "no_pressure"), (["capacity_normalized_pressure"], "pressure")]:
            fit, support = fit_absorbed_fe(recovery, outcome, ["exposure_recovery"] + controls)
            pooled_rows.append(coefficient_row(fit, support, outcome, f"{label}_{suffix}", "exposure_recovery"))
    weekly_rows, pretrend_rows = [], []
    event_terms = []
    for week in RECOVERY_WEEKS:
        term = f"exposure_week_{week}"
        recovery[term] = recovery.log1p_unique_evacuees * recovery.week.eq(week).astype(int)
        event_terms.append(term)
    for outcome, _ in specs[:-1]:
        fit, support = fit_absorbed_fe(recovery, outcome, event_terms + ["capacity_normalized_pressure"])
        for week, term in zip(RECOVERY_WEEKS, event_terms):
            row = coefficient_row(fit, support, outcome, "week_specific_pressure", term)
            row["week"] = week
            weekly_rows.append(row)
        if pretrend:
            baseline = panel.loc[panel.week.isin(BASELINE_WEEKS)].copy()
            terms = []
            for week in BASELINE_WEEKS[:-1]:
                term = f"baseline_exposure_week_{week}_vs_5"
                baseline[term] = baseline.log1p_unique_evacuees * baseline.week.eq(week).astype(int)
                terms.append(term)
            fit_pre, support_pre = fit_absorbed_fe(baseline, outcome, terms + ["capacity_normalized_pressure"])
            restriction = np.zeros((len(terms), len(fit_pre.params)))
            for row_index, term in enumerate(terms):
                restriction[row_index, list(fit_pre.params.index).index(term)] = 1
            test = fit_pre.wald_test(restriction)
            pretrend_rows.append(dict(outcome=outcome, joint_baseline_interaction_p_value=float(test.pvalue), **support_pre))
    return pd.DataFrame(pooled_rows), pd.DataFrame(weekly_rows), pd.DataFrame(pretrend_rows)


def descriptive_tables(panel):
    overall = panel.groupby("week", as_index=False).all_visits.sum()
    baseline = overall.loc[overall.week.le(5), "all_visits"].mean()
    overall["normal_index"] = 100 * overall.all_visits / baseline
    parts = []
    for definition, group in [("any_evacuee", "shared_any"), ("at_least_3", "shared_ge3")]:
        weekly = panel.loc[panel.host_active].groupby(["week", group], as_index=False).agg(
            total_host_dwell=("host_total_dwell", "sum"), total_host_visits=("host_visits", "sum"),
            active_pois=("place_id", "nunique")).rename(columns={group: "shared"})
        weekly["mean_host_dwell_per_active_poi"] = weekly.total_host_dwell / weekly.active_pois
        weekly["mean_host_dwell_per_visit"] = weekly.total_host_dwell / weekly.total_host_visits
        weekly["visits_per_active_poi"] = weekly.total_host_visits / weekly.active_pois
        for metric in ["mean_host_dwell_per_active_poi", "mean_host_dwell_per_visit", "visits_per_active_poi"]:
            base = weekly.loc[weekly.week.le(5)].groupby("shared")[metric].mean()
            weekly[f"{metric}_change_pct"] = 100 * (weekly[metric] / weekly.shared.map(base) - 1)
        weekly["definition"] = definition
        parts.append(weekly)
    active = pd.concat(parts, ignore_index=True)
    active["phase"] = np.select([active.week.isin(ACUTE_WEEKS), active.week.isin(RECOVERY_WEEKS)],
                                ["acute", "recovery"], default="other")
    columns = [c for c in active if c.endswith("_change_pct")]
    summary = active.loc[active.phase.ne("other")].groupby(["definition", "shared", "phase"], as_index=False)[columns].mean()
    # Equal-POI comparisons retain zero visits in the balanced panel. Integer
    # cross-products distinguish exact equality from an increase without rounding.
    wide = panel.pivot(index="place_id", columns="week", values="all_visits").astype("int64")
    baseline_sum = wide.loc[:, 0:5].sum(axis=1)
    if not baseline_sum.gt(0).all():
        raise ValueError("All baseline-selected POIs must have positive baseline visits")
    shared = panel.groupby("place_id").shared_ge3.first().reindex(wide.index)
    rows, increases = [], {}
    for phase, weeks in [("acute", ACUTE_WEEKS), ("recovery", RECOVERY_WEEKS)]:
        increased = 6 * wide[weeks].sum(axis=1) > len(weeks) * baseline_sum
        increases[phase] = increased
        any_week = wide[weeks].mul(6).gt(baseline_sum, axis=0).any(axis=1)
        for name, keep in [("All POIs", pd.Series(True, index=wide.index)), ("Shared POIs", shared), ("Below-threshold POIs", ~shared)]:
            rows.append(dict(group=name, period=phase, n_pois=int(keep.sum()),
                             pct_increased=100 * increased[keep].mean(),
                             pct_any_week_increased=100 * any_week[keep].mean()))
    joint = []
    for name, keep in [("All POIs", pd.Series(True, index=wide.index)), ("Shared POIs", shared), ("Below-threshold POIs", ~shared)]:
        joint.append(dict(group=name, n_pois=int(keep.sum()), pct_increased_both_period_means=100 * (increases["acute"] & increases["recovery"])[keep].mean()))
    return overall, active, summary, pd.DataFrame(rows), pd.DataFrame(joint)


def plot_adjusted(weekly, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    for outcome, label, color, offset in [
        ("log_host_visits_positive", "Host visits", "#0072b2", -.05),
        ("log_host_total_dwell_positive", "Host dwell time", "#cc79a7", .05),
    ]:
        q = weekly.loc[weekly.outcome.eq(outcome)].sort_values("week")
        ax.errorbar(q.week + offset, q.coefficient,
                    yerr=[q.coefficient - q.ci_lower, q.ci_upper - q.coefficient],
                    fmt="o-", capsize=3, color=color, label=label)
    ax.axhline(0, color=".45", linewidth=.8)
    ax.set(xticks=RECOVERY_WEEKS, xlabel="Recovery week",
           ylabel="Pressure-adjusted exposure coefficient (95% CI)")
    ax.legend(frameon=False)
    ax.text(.98, .03, "Outcome-specific positive support", transform=ax.transAxes,
            ha="right", fontsize=8)
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(output / f"comment2_C_adjusted_host_outcomes.{suffix}", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", required=True, type=Path, help="Existing balanced poi_week_pressure_panel.csv[.gz]")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pretrend", action="store_true", help="Also run the six baseline-interaction joint tests")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    panel = load_panel(args.panel)
    pooled, weekly, pretrend = estimate_models(panel, args.pretrend)
    overall, active, summary, local, joint = descriptive_tables(panel)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, table in {
        "pooled_margin_models.csv": pooled, "weekly_margin_models.csv": weekly,
        "poi_level_all_visits_weekly.csv": overall,
        "active_poi_metrics_weekly.csv": active, "active_poi_visit_dwell_summary.csv": summary,
        "localized_visitation_summary.csv": local, "localized_visitation_joint.csv": joint,
    }.items():
        table.to_csv(args.output / name, index=False)
    if args.pretrend:
        pretrend.to_csv(args.output / "dwell_margin_pretrend_tests.csv", index=False)
    q = pooled.set_index("specification")
    names = ["conditional_positive_log_total_dwell_pressure", "conditional_positive_log_visits_pressure", "conditional_log_mean_dwell_per_visit_pressure"]
    identity_error = q.loc[names[0], "coefficient"] - q.loc[names[1], "coefficient"] - q.loc[names[2], "coefficient"]
    audit = dict(input_file=args.panel.name, input_rows=len(panel), n_pois=panel.place_id.nunique(),
                 baseline_weeks=BASELINE_WEEKS, recovery_weeks=RECOVERY_WEEKS,
                 pooled_models=len(pooled), weekly_coefficients=len(weekly), pretrend_run=args.pretrend,
                 conditional_log_identity_error=float(identity_error),
                 identity_samples_equal=bool(q.loc[names, "n_obs"].nunique() == 1),
                 input_stage="Saved baseline-selected balanced POI-week panel; raw processing not rerun",
                 inference="Two-way POI/week fixed effects; POI clusters; absorbed-DF correction; cluster-t intervals")
    (args.output / "analysis_summary.json").write_text(json.dumps(audit, indent=2) + "\n")
    if not args.no_figures:
        plot_adjusted(weekly, args.output)
    print(f"Saved {len(pooled)} pooled models and {len(weekly)} weekly coefficients to {args.output}")


if __name__ == "__main__":
    main()
