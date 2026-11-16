"""Summarize Ian recovery in social mixing and host activity.

Required input: mixing_all_specifications.csv, with specification identifiers,
week_start_date, mean_individual_seg and pct_change_mean_individual_seg.
Optional host_dwell_all_by_received_weekly.txt adds fixed shared/comparison
POI dwell per visit. Optional fixed_encounter_weekly_source_label_conflict.csv
adds the fixed-profile Encounter Index and Host Shared Ratio. The saved SIMPLE
encounter export used a three-evacuee threshold despite an earlier label conflict.

All analyses select Top-100 receiving communities and strict-stayer hosts.
Mixing uses weekly eligible samples; aggregate dwell and overlap retain their
upstream balanced-host construction. These scripts summarize saved estimates.
"""
from pathlib import Path
import argparse
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from common.ian_metrics import (EVENT_START, NORMALIZATION, read_table,
    weekly_check, select_mixing, period_values, save_audit, plotting, style_time, save_plot)


COHORTS = [("host", "strict_stayer", "Hosts", "#3D6586"),
           ("evacuee", "evacuee", "Evacuees", "#B77740"),
           ("comparison", "non_host_non_evacuee", "Other visitors", "#879298")]


def analyze_mixing(frame):
    cohorts, thresholds, summaries = [], [], {"cohorts": {}, "thresholds": {}}
    for cohort, rule, _, _ in COHORTS:
        q = select_mixing(frame, cohort, rule)
        cohorts.append(q[["cohort", "week_start_date", "M", "normalized_change"]])
        summaries["cohorts"][cohort] = period_values(q)
    for threshold in [1, 3, 5, 10]:
        q = select_mixing(frame, threshold=threshold)
        thresholds.append(q[["min_evac_devices", "week_start_date", "M", "normalized_change"]])
        summaries["thresholds"][str(threshold)] = period_values(q)
    return pd.concat(cohorts), pd.concat(thresholds), summaries


def analyze_dwell(frame):
    q = weekly_check(frame.loc[(frame.top_n == 100) &
        (frame.host_definition == "all_by_received") & (frame.host_rule == "strict_stayer") &
        (frame.evac_threshold == 3) & (frame.shared_scheme == "fixed")])
    rows = []
    for group in ["shared", "nonshared"]:
        average = q[f"{group}_host_dwell"] / q[f"{group}_host_visits"]
        np.testing.assert_allclose(average, q[f"{group}_avg_dwell"])
        baseline = average.loc[q.week_start_date.lt(EVENT_START)].mean()
        change = 100 * (average / baseline - 1)
        np.testing.assert_allclose(change, q[f"{group}_avg_change_pct"], atol=1e-7)
        rows.append(pd.DataFrame({"group": group, "week_start_date": q.week_start_date,
            "dwell_per_visit": average, "baseline_dwell_per_visit": baseline,
            "change_pct": q[f"{group}_avg_change_pct"]}))
    return pd.concat(rows)


def analyze_overlap(frame):
    q = weekly_check(frame.loc[(frame.top_n == 100) &
        (frame.host_definition == "all_by_received") & (frame.host_rule == "strict_stayer")])
    assert q.n_evac_places.nunique() == 1 and q.total_evac_dwell.nunique() == 1
    rows = []
    for metric in ["encounter_index", "host_shared_ratio"]:
        baseline = q.loc[q.week_start_date.lt(EVENT_START), metric].mean()
        change = 100 * (q[metric] / baseline - 1)
        np.testing.assert_allclose(change, q[f"{metric}_change_pct"], atol=1e-7)
        rows.append(pd.DataFrame({"metric": metric, "week_start_date": q.week_start_date,
            "value": q[metric], "baseline": baseline, "change_pct": q[f"{metric}_change_pct"]}))
    return pd.concat(rows)


def plot_recovery(cohorts, thresholds, dwell, overlap, output):
    plt = plotting()
    fig, ax = plt.subplots(figsize=(8, 3.9))
    for cohort, _, label, color in COHORTS:
        q = cohorts.loc[cohorts.cohort.eq(cohort)].sort_values("week_start_date")
        ax.plot(q.week_start_date, q.normalized_change, color=color, marker="o",
                markersize=3.5, linewidth=1.8, label=label)
    style_time(ax, "Normalized change in experienced\nsocial mixing (%)")
    ax.set_title("Hurricane Ian · Top-100 host communities")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    save_plot(fig, output, "mixing_recovery")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.9))
    for threshold, color, marker in zip([1, 3, 5, 10],
        ["#879298", "#3D6586", "#268579", "#B77740"], ["o", "s", "^", "D"]):
        q = thresholds.loc[thresholds.min_evac_devices.eq(threshold)].sort_values("week_start_date")
        ax.plot(q.week_start_date, q.normalized_change, color=color, marker=marker,
                markersize=3.5, linewidth=1.8, label=f"At least {threshold} evacuee{'s' if threshold != 1 else ''}")
    style_time(ax, "Normalized change in experienced\nsocial mixing (%)")
    ax.set_title("Top-100 hosts · fixed location-selection thresholds")
    ax.legend(frameon=False, ncol=2, fontsize=9)
    fig.tight_layout()
    save_plot(fig, output, "location_threshold_sensitivity")
    plt.close(fig)
    if dwell is not None:
        fig, ax = plt.subplots(figsize=(8, 3.9))
        for group, label, color in [("shared", "Shared POIs", "#268579"),
                                    ("nonshared", "Comparison POIs", "#B77740")]:
            q = dwell.loc[dwell.group.eq(group)].sort_values("week_start_date")
            ax.plot(q.week_start_date, q.change_pct, marker="o", color=color, label=label)
        style_time(ax, "Host dwell time per visit: change (%)")
        ax.legend(frameon=False)
        fig.tight_layout()
        save_plot(fig, output, "host_dwell_recovery")
        plt.close(fig)
    if overlap is not None:
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        for ax, metric, title in zip(axes, ["encounter_index", "host_shared_ratio"],
                                     ["Encounter Index", "Host Shared Ratio"]):
            q = overlap.loc[overlap.metric.eq(metric)].sort_values("week_start_date")
            ax.plot(q.week_start_date, q.change_pct, marker="o", color="#268579")
            style_time(ax, "Change from baseline (%)")
            ax.set_title(title)
        fig.tight_layout()
        save_plot(fig, output, "host_activity_overlap")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixing", type=Path, required=True)
    parser.add_argument("--dwell", type=Path, help="Optional saved host-dwell aggregate export")
    parser.add_argument("--overlap", type=Path, help="Optional fixed-profile aggregate overlap export")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    cohorts, thresholds, summary = analyze_mixing(read_table(args.mixing))
    dwell = analyze_dwell(read_table(args.dwell)) if args.dwell else None
    overlap = analyze_overlap(read_table(args.overlap)) if args.overlap else None
    args.output.mkdir(parents=True, exist_ok=True)
    cohorts.to_csv(args.output / "mixing_recovery_weekly.csv", index=False)
    thresholds.to_csv(args.output / "location_thresholds_weekly.csv", index=False)
    for data, filename, group_column in [(dwell, "host_dwell_weekly", "group"),
                                          (overlap, "host_overlap_weekly", "metric")]:
        if data is not None:
            data.to_csv(args.output / f"{filename}.csv", index=False)
            summary[filename] = {name: period_values(q, "change_pct") for name, q in data.groupby(group_column)}
    summary["mixing_normalization"] = NORMALIZATION
    summary["dwell_overlap_normalization"] = "100*(X/X0-1)"
    save_audit(args.output, "recovery_summary.json", {"mixing": args.mixing,
        "dwell": args.dwell, "overlap": args.overlap}, summary)
    if args.plots:
        plot_recovery(cohorts, thresholds, dwell, overlap, args.output)
    print(f"Wrote recovery summaries to {args.output}")


if __name__ == "__main__":
    main()
