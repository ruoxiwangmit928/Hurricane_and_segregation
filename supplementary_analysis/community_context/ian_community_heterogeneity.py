"""Analyze heterogeneity in Ian host-community recovery and characteristics.

Inputs are the tab-delimited ian cbg weekly metric.txt and ian cbg feature.txt
exports. Retain their original Top-200 exposure measurements; rank the 200
communities by n_received_evacuees descending and host_cbg ascending, then
select Top 100. No location profiles or visitor composition are re-estimated.
The original three-metric rule identifies 12 joint-response and 88 other CBGs.

Outputs summarize normalized social mixing, Encounter Index, Host Shared
Ratio, community characteristics and paired origin/host income. Weekly
changes are calculated within CBGs before equally weighted group summaries.
"""
from pathlib import Path
import argparse
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from common.ian_metrics import (WEEKS, EVENT_START, POST, LATE, NORMALIZATION,
    read_table, normalized_from_s, classify_joint_response, save_audit,
    plotting, style_time, save_plot)

CHARACTERISTICS = ["host_cbg_income", "mean_origin_income", "host_minus_origin_income",
    "mean_abs_income_gap", "n_received_evacuees", "received_per_1000_residents",
    "mean_displacement_days", "share_right_censored"]
OVERLAP_METRICS = ["pct_change_encounter_index", "pct_change_shared_dwell_ratio"]


def analyze_communities(weekly, features):
    assert len(features) == features.host_cbg.nunique() == 200
    roster = features.sort_values(["n_received_evacuees", "host_cbg"],
                                  ascending=[False, True]).iloc[:100].copy()
    w = weekly.loc[weekly.host_definition.eq("all_by_received") &
                   weekly.host_rule.eq("strict_stayer")].copy()
    assert w.top_n.eq(200).all()
    assert w.min_evac_devices.dropna().eq(3).all()
    assert not w.duplicated(["host_cbg", "week_start_date"]).any()
    assert set(w.week_start_date) == set(WEEKS)
    assert w.groupby("host_cbg").week_start_date.nunique().eq(12).all()
    w = w.loc[w.host_cbg.isin(roster.host_cbg)].copy()
    assert w.host_cbg.nunique() == 100
    classification = classify_joint_response(w)
    assert classification.eligible.sum() == 100 and classification.selected.sum() == 12
    w["selected"] = w.host_cbg.map(classification.selected)
    w["M"] = 1 - w.mean_individual_seg
    w["normalized_change"] = -w.pct_change_mean_individual_seg
    baseline_s = w.loc[w.week_start_date.lt(EVENT_START)].groupby("host_cbg").mean_individual_seg.mean()
    np.testing.assert_allclose(w.normalized_change,
        normalized_from_s(w.mean_individual_seg, w.host_cbg.map(baseline_s)), atol=1e-8, equal_nan=True)
    np.testing.assert_allclose(w.loc[w.week_start_date.lt(EVENT_START)].groupby("host_cbg").normalized_change.mean(),
        0, atol=1e-10)
    mixing = w.groupby(["selected", "week_start_date"], as_index=False).agg(
        mean=("normalized_change", "mean"), median=("normalized_change", "median"),
        n_cbgs=("normalized_change", "count"))
    assert mixing.n_cbgs.eq(mixing.selected.map({True: 12, False: 88})).all()
    for raw, baseline, change in [
        ("encounter_index", "baseline_encounter_index", OVERLAP_METRICS[0]),
        ("host_shared_dwell_ratio", "baseline_shared_dwell_ratio", OVERLAP_METRICS[1])]:
        reconstructed = w.loc[w.week_start_date.lt(EVENT_START)].groupby("host_cbg")[raw].mean()
        np.testing.assert_allclose(w.host_cbg.map(reconstructed), w[baseline], atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(100 * (w[raw] / w[baseline] - 1), w[change], atol=1e-8, equal_nan=True)
    overlap_rows = []
    for (selected, date), q in w.groupby(["selected", "week_start_date"]):
        for metric in OVERLAP_METRICS:
            values = q[metric].dropna()
            overlap_rows.append({"selected": bool(selected), "week_start_date": date, "metric": metric,
                "n_cbgs": len(values), "mean": float(values.mean()), "median": float(values.median())})
    overlap = pd.DataFrame(overlap_rows)
    f = roster.set_index("host_cbg").copy()
    f["host_minus_origin_income"] = f.host_cbg_income - f.mean_origin_income
    f["selected"] = classification.selected
    f = f.loc[classification.index[classification.eligible]]
    characteristic_rows = []
    for selected, q in f.groupby("selected"):
        for metric in CHARACTERISTICS:
            values = q[metric].dropna()
            characteristic_rows.append({"selected": bool(selected), "metric": metric, "n_cbgs": len(values),
                "mean": float(values.mean()), "median": float(values.median())})
    characteristics = pd.DataFrame(characteristic_rows)
    paired_income = roster[["host_cbg_income", "mean_origin_income"]].dropna()
    gap = paired_income.host_cbg_income - paired_income.mean_origin_income
    income_summary = {"n_paired_cbgs": len(paired_income), "n_host_higher": int(gap.gt(0).sum()),
        "pct_host_higher": float(100 * gap.gt(0).mean()), "mean_host_minus_origin": float(gap.mean()),
        "median_host_minus_origin": float(gap.median())}
    curves = pd.concat([mixing.assign(metric="normalized_mixing_change"), overlap], ignore_index=True)
    period_rows = []
    for period, dates in [("event_three_weeks", WEEKS[7:10]), ("post_four_weeks", POST),
                          ("final_two_weeks", LATE)]:
        for (selected, metric), q in curves.loc[curves.week_start_date.isin(dates)].groupby(["selected", "metric"]):
            period_rows.append({"period": period, "selected": bool(selected), "metric": metric,
                "n_cbgs": 12 if selected else 88, "mean_of_weekly_means": float(q["mean"].mean()),
                "mean_of_weekly_medians": float(q["median"].mean())})
    return mixing, overlap, characteristics, pd.DataFrame(period_rows), income_summary, paired_income


def plot_communities(mixing, overlap, paired_income, output):
    plt = plotting()
    colors = {True: "#25836C", False: "#6A7C90"}
    fig, ax = plt.subplots(figsize=(8, 5.1), layout="constrained")
    style_time(ax, "Normalized change in experienced\nsocial mixing (%)")
    for selected in [True, False]:
        q = mixing.loc[mixing.selected.eq(selected)].sort_values("week_start_date")
        label = ("Joint-response" if selected else "Other") + f" CBGs (n={12 if selected else 88})"
        for statistic, linestyle, marker in [("mean", "-", "o"), ("median", "--", "s")]:
            ax.plot(q.week_start_date, q[statistic], color=colors[selected], ls=linestyle, marker=marker,
                markerfacecolor=colors[selected] if statistic == "mean" else "white", ms=3.5,
                lw=1.6, label=label + ": " + statistic)
    ax.set_title("Experienced social mixing: Top-100 host communities", loc="left", pad=10)
    ax.legend(frameon=False, ncol=2, fontsize=9, loc="upper center", bbox_to_anchor=(.5, -.25))
    save_plot(fig, output, "community_mixing_recovery")
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6.7), sharex=True, layout="constrained")
    for ax, metric, title in zip(axes, OVERLAP_METRICS, ["Encounter Index", "Host Shared Ratio"]):
        for selected in [True, False]:
            q = overlap.loc[overlap.selected.eq(selected) & overlap.metric.eq(metric)].sort_values("week_start_date")
            label = ("Joint-response" if selected else "Other") + f" CBGs (n={12 if selected else 88})"
            ax.plot(q.week_start_date, q["median"], color=colors[selected], marker="o", ms=3.5, lw=1.6, label=label)
        style_time(ax, "Median change from baseline (%)")
        ax.set_title(title, loc="left", pad=10)
    axes[0].set_xlabel("")
    axes[0].legend(frameon=False, ncol=2, fontsize=9)
    save_plot(fig, output, "community_activity_overlap")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.6, 5.2), layout="constrained")
    ax.scatter(paired_income.mean_origin_income, paired_income.host_cbg_income,
               s=32, alpha=.72, color="#3D6586", linewidths=.4, edgecolors="white")
    maximum = np.ceil(paired_income.max().max() / 10000) * 10000
    ax.plot([0, maximum], [0, maximum], "--", color="#8B949B", lw=1)
    ax.set(xlim=(0, maximum), ylim=(0, maximum), aspect="equal",
        xlabel="Mean evacuee origin-neighborhood income ($)",
        ylabel="Host-neighborhood median income ($)")
    save_plot(fig, output, "origin_host_income_comparison")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weekly", type=Path, required=True, help="Original Top-200 CBG-week export")
    parser.add_argument("--features", type=Path, required=True, help="Original 200-CBG feature export")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    mixing, overlap, characteristics, periods, income, paired = analyze_communities(
        read_table(args.weekly), read_table(args.features))
    args.output.mkdir(parents=True, exist_ok=True)
    for name, table in [("community_mixing_weekly", mixing), ("community_overlap_weekly", overlap),
                        ("community_characteristics", characteristics), ("community_periods", periods)]:
        table.to_csv(args.output / f"{name}.csv", index=False)
    pd.DataFrame([income]).to_csv(args.output / "origin_host_income_summary.csv", index=False)
    save_audit(args.output, "community_summary.json", {"weekly": args.weekly, "features": args.features},
        {"normalization": NORMALIZATION, "source_measurement_top_n": 200, "display_community_top_n": 100,
         "group_counts": {"joint_response": 12, "other": 88}, "income_comparison": income,
         "aggregation": "Normalize within each CBG before equally weighted cross-CBG summaries"})
    if args.plots:
        plot_communities(mixing, overlap, paired, args.output)
    print(f"Wrote community summaries to {args.output}")


if __name__ == "__main__":
    main()
