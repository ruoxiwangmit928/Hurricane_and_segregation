"""Harvey host overlap and shared-location dwell trajectories (main Figure 4b/c).

Run:
  python 1_encounter_index.py --evacuee-visits evacuees_poi_visits.csv \
      --host-activity host_poi_week_activity.csv --output results/host_overlap

Inputs are prepared records, not raw GPS: evacuee visits must contain device_id,
place_id, date, start_date, end_date, stay_time; host activity must have week,
place_id, host_total_dwell with one row per host-active POI/week for weeks 0--11.
Keep the original host roster and upstream visit filters when preparing it.

Main panel b: BOTH Encounter Index and Host Shared Ratio retain the complete
fixed acute-period profile, without a three-person filter. They measure spatial
alignment/potential overlap, not contemporaneous interpersonal encounters.
Panel c: shared locations require >=3 distinct evacuees inside their recorded
episodes during Aug25--Sep7 inclusive. Dwell is mean total host dwell per active
POI in each group/week, expressed as percent change from weeks 0--5.
The mixing complement-baseline formula does not apply to dwell or overlap.
Outputs are aggregate tables and separate figures; raw records are not exported.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from shared_location_reference import (build_reference, read_host_activity,
    plotting, save_plot, WEEK0, REFERENCE_START, REFERENCE_END, BASELINE_WEEKS, COLORS)


def calculate(profile, shared_ids, host):
    total_evacuee_dwell = profile.evacuee_dwell.sum()
    overlap, dwell = [], []
    for week, group in host.groupby("week", sort=True):
        total = group.host_total_dwell.sum()
        if total <= 0:
            raise ValueError(f"Nonpositive host dwell in week {week}")
        aligned = group.place_id.map(profile.evacuee_dwell).fillna(0)
        full_shared = group.place_id.isin(profile.index)
        overlap.append(dict(week=int(week),
            encounter_index=float((group.host_total_dwell * aligned).sum() / (total * total_evacuee_dwell)),
            host_shared_ratio=float(group.loc[full_shared, "host_total_dwell"].sum() / total)))
        ge3 = group.place_id.isin(shared_ids)
        if not ge3.any() or ge3.all():
            raise ValueError(f"Missing comparison-group support in week {week}")
        dwell.append(dict(week=int(week),
            host_dwell_shared=float(group.loc[ge3, "host_total_dwell"].mean()),
            host_dwell_nonshared=float(group.loc[~ge3, "host_total_dwell"].mean()),
            n_active_shared=int(ge3.sum()), n_active_nonshared=int((~ge3).sum())))
    overlap, dwell = pd.DataFrame(overlap), pd.DataFrame(dwell)
    for col in ["host_dwell_shared", "host_dwell_nonshared"]:
        baseline = dwell.loc[dwell.week.isin(BASELINE_WEEKS), col].mean()
        if baseline <= 0:
            raise ValueError(f"Nonpositive pre-event {col}")
        dwell[f"change_{col}_pct"] = 100 * (dwell[col] / baseline - 1)
    for frame in [overlap, dwell]:
        frame["week_start"] = WEEK0 + pd.to_timedelta(7 * frame.week, unit="D")
    return overlap, dwell


def draw(overlap, dwell, output):
    plt = plotting()
    import matplotlib.dates as mdates
    from matplotlib.ticker import FormatStrFormatter
    def time_axis(ax):
        ax.axvspan(REFERENCE_START, REFERENCE_END + pd.Timedelta(days=1), color="0.88", zorder=0)
        ax.set_xticks(overlap.week_start)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.tick_params(axis="x", rotation=45, labelsize=9)
        ax.set_xlabel("Week starting in 2017")
        ax.margins(x=.02)
    fig, left = plt.subplots(figsize=(6.7, 3.8))
    right = left.twinx()
    for ax, col, color, label in [(left, "encounter_index", "#3776ab", "Encounter Index"),
                                  (right, "host_shared_ratio", "#ed8b32", "Host Shared Ratio")]:
        ax.plot(overlap.week_start, overlap[col], "o-", color=color, markerfacecolor="white", ms=4.2, lw=1.5)
        ax.axhline(overlap.loc[overlap.week.isin(BASELINE_WEEKS), col].mean(), color=color, ls="--", lw=.8)
        ax.set_ylabel(label, color=color)
        ax.tick_params(axis="y", colors=color)
    left.yaxis.set_major_formatter(FormatStrFormatter("%.5f"))
    right.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    time_axis(left)
    fig.subplots_adjust(left=.16, right=.86, bottom=.24, top=.95)
    save_plot(fig, output, "encounter_index")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.7, 3.8))
    for col, color, label in [("change_host_dwell_shared_pct", COLORS[0], "Shared locations"),
                               ("change_host_dwell_nonshared_pct", COLORS[1], "Non-shared locations")]:
        ax.plot(dwell.week_start, dwell[col], "o-", color=color, label=label,
                markerfacecolor="white", ms=4.2, lw=1.5)
    ax.axhline(0, color="0.6", ls="--", lw=.8)
    ax.set_ylabel("Host dwell time change (%)")
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    time_axis(ax)
    fig.subplots_adjust(left=.16, right=.96, bottom=.24, top=.95)
    save_plot(fig, output, "shared_stay_change")
    plt.close(fig)


def run(evacuee_visits, host_activity, output, figures=True):
    profile, shared, audit = build_reference(evacuee_visits)
    host = read_host_activity(host_activity)
    overlap, dwell = calculate(profile, shared, host)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    overlap.to_csv(output / "encounter_index_weekly.csv", index=False)
    dwell.to_csv(output / "shared_dwell_weekly.csv", index=False)
    audit.update(panel_b_reference="Complete acute-period reference for BOTH EI and HSR",
        panel_c_reference="At least 3 distinct evacuees; fixed across weeks",
        weekly_dwell_denominator="Number of host-active POIs within each group/week",
        baseline_weeks=BASELINE_WEEKS, dwell_change="100*(dwell/dwell0-1)",
        recovery_weeks9_11=dwell.loc[dwell.week.ge(9),
            ["change_host_dwell_shared_pct", "change_host_dwell_nonshared_pct"]].mean().to_dict())
    (output / "analysis_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if figures: draw(overlap, dwell, output)
    return overlap, dwell, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evacuee-visits", required=True, type=Path)
    parser.add_argument("--host-activity", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    run(args.evacuee_visits, args.host_activity, args.output, not args.no_figures)
    print(f"Saved aggregate tables and requested figures to {args.output}")


if __name__ == "__main__":
    main()
