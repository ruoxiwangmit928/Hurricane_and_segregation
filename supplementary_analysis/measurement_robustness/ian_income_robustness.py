"""Assess Ian host-mixing robustness to income-distribution assignments.

Input: 05_weekly_results_all_imputations.txt, a tab-delimited aggregate export
with method, imputation, cohort, host_definition, host_rule, top_n,
week_start_date, mean_individual_seg and n_person_weeks. Uses the Top-100
strict-stayer host sample. One median reference and 100 assignments for each
alternative are required. Each draw is normalized before averaging draws.

This summarizes existing income assignments; it does not generate new incomes
or rerun the authorized upstream GE3EVAC visitor-composition pipeline.
"""
from pathlib import Path
import argparse
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from common.ian_metrics import (EVENT_START, NORMALIZATION, normalized_from_s,
    read_table, period_values, save_audit, plotting, style_time, save_plot)


def analyze_income_assignments(frame):
    ses = frame.loc[(frame.top_n == 100) & (frame.cohort == "host") &
        (frame.host_definition == "all_by_received") & (frame.host_rule == "strict_stayer")].copy()
    keys = ["method", "imputation"]
    assert not ses.duplicated(keys + ["week_start_date"]).any()
    assert ses.groupby(keys).size().eq(12).all()
    assert ses.groupby("method").imputation.nunique().to_dict() == {
        "median": 1, "uniform_bin": 100, "lognormal": 100}
    assert ses.groupby("week_start_date").n_person_weeks.nunique().eq(1).all()
    ses["M"] = 1 - ses.mean_individual_seg
    base_s = ses.loc[ses.week_start_date.lt(EVENT_START)].groupby(keys).mean_individual_seg.mean()
    ses = ses.join(base_s.rename("baseline_s"), on=keys)
    ses["normalized_change"] = normalized_from_s(ses.mean_individual_seg, ses.baseline_s)
    ses["baseline_M"] = 1 - ses.baseline_s
    np.testing.assert_allclose(ses.normalized_change,
        100 * (ses.M - ses.baseline_M) / (1 - ses.baseline_M), atol=1e-10)
    np.testing.assert_allclose(ses.loc[ses.week_start_date.lt(EVENT_START)].groupby(keys).normalized_change.mean(),
        0, atol=1e-10)
    weekly = ses.groupby(["method", "week_start_date"], as_index=False).agg(
        M=("M", "mean"), normalized_change=("normalized_change", "mean"),
        baseline_M=("baseline_M", "mean"), n=("n_person_weeks", "first"))
    summaries = {method: {"baseline_M": float(q.baseline_M.iloc[0]), **period_values(q)}
                 for method, q in weekly.groupby("method")}
    return weekly, summaries


def plot_income_assignments(weekly, output):
    plt = plotting()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
    for method, label, color, marker in [
        ("median", "CBG median", "#30383E", "o"),
        ("uniform_bin", "Uniform within income bin", "#268579", "s"),
        ("lognormal", "Log-normal income", "#B77740", "^")]:
        q = weekly.loc[weekly.method.eq(method)].sort_values("week_start_date")
        for ax, column in zip(axes, ["M", "normalized_change"]):
            ax.plot(q.week_start_date, q[column], label=label, color=color, marker=marker,
                    markersize=3.5, linewidth=1.8)
    style_time(axes[0], "Experienced social mixing", zero=False)
    style_time(axes[1], "Normalized change in experienced\nsocial mixing (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(.5, 1.045), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .92))
    save_plot(fig, output, "income_assignment_sensitivity")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weekly-assignments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plots", action="store_true", help="Also write PNG/PDF plots")
    args = parser.parse_args()
    weekly, summaries = analyze_income_assignments(read_table(args.weekly_assignments))
    args.output.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(args.output / "income_assignment_weekly.csv", index=False)
    save_audit(args.output, "income_assignment_summary.json", {"weekly_assignments": args.weekly_assignments},
        {"normalization": NORMALIZATION, "normalization_before_draw_averaging": True,
         "sample": "Top-100 all_by_received strict-stayer hosts", "methods": summaries})
    if args.plots:
        plot_income_assignments(weekly, args.output)
    print(f"Wrote income-assignment summaries to {args.output}")


if __name__ == "__main__":
    main()
