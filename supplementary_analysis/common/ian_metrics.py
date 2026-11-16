"""Shared measurement definitions for Ian supplementary analyses.

Inputs are saved aggregate or CBG-level exports, never raw device records.
Mixing changes use 100*(M-M0)/(1-M0), with M=1-S. Overlap and dwell
changes retain 100*(X/X0-1). The baseline spans seven pre-event weeks.
"""
from pathlib import Path
import hashlib
import json
import os
import tempfile

import numpy as np
import pandas as pd

WEEKS = pd.date_range("2022-08-05", periods=12, freq="7D")
EVENT_START = pd.Timestamp("2022-09-23")
EVENT_END = pd.Timestamp("2022-10-11")
POST = pd.to_datetime(["2022-09-30", "2022-10-07", "2022-10-14", "2022-10-21"])
LATE = POST[-2:]
NORMALIZATION = "100*(M-M0)/(1-M0), with M=1-S"


def read_table(path):
    path = Path(path)
    frame = pd.read_csv(path, sep="\t" if path.suffix.lower() in (".txt", ".tsv") else ",",
                        dtype={"host_cbg": str, "origin_cbg": str})
    if "week_start_date" in frame:
        frame["week_start_date"] = pd.to_datetime(frame.week_start_date)
    return frame


def weekly_check(frame):
    frame = frame.sort_values("week_start_date").copy()
    assert len(frame) == frame.week_start_date.nunique() == 12
    assert list(frame.week_start_date) == list(WEEKS)
    return frame


def normalized_from_s(s, baseline_s):
    """Equivalent to 100*(M-M0)/(1-M0); never divide by baseline M."""
    return -100 * (s / baseline_s - 1)


def select_mixing(frame, cohort="host", rule="strict_stayer", threshold=3):
    q = weekly_check(frame.loc[(frame.top_n == 100) & (frame.cohort == cohort) &
        (frame.min_evac_devices == threshold) & (frame.host_rule == rule) &
        (frame.host_definition == "all_by_received")])
    q["M"] = 1 - q.mean_individual_seg
    baseline_s = q.loc[q.week_start_date.lt(EVENT_START), "mean_individual_seg"].mean()
    q["normalized_change"] = normalized_from_s(q.mean_individual_seg, baseline_s)
    np.testing.assert_allclose(q.normalized_change, -q.pct_change_mean_individual_seg, atol=1e-8)
    np.testing.assert_allclose(q.normalized_change, 100 * (q.M - (1-baseline_s)) / baseline_s, atol=1e-10)
    return q


def classify_joint_response(frame):
    """Original joint rule: two valid weeks per metric; no fixed subgroup size."""
    a = pd.DataFrame(index=sorted(frame.host_cbg.unique()))
    a["eligible"] = True
    for name, col, positive in [
        ("mixing", "pct_change_mean_individual_seg", True),
        ("encounter", "pct_change_encounter_index", False),
        ("shared", "pct_change_shared_dwell_ratio", False)]:
        p = frame.pivot(index="host_cbg", columns="week_start_date", values=col).reindex(columns=POST)
        count = p.count(axis=1)
        if positive:
            condition = (p.ge(5).sum(axis=1).ge(2) | p.mean(axis=1).ge(5) | p[LATE].mean(axis=1).ge(5))
        else:
            condition = (p.le(-5).sum(axis=1).ge(2) | p.mean(axis=1).le(-5) | p[LATE].mean(axis=1).le(-5))
        a[name + "_rule"] = count.ge(2) & condition
        a["eligible"] &= count.ge(2)
    a["selected"] = a.mixing_rule & a.encounter_rule & a.shared_rule
    return a


def period_values(frame, column="normalized_change"):
    return {
        "sep23": float(frame.loc[frame.week_start_date.eq("2022-09-23"), column].iloc[0]),
        "sep30": float(frame.loc[frame.week_start_date.eq("2022-09-30"), column].iloc[0]),
        "oct14": float(frame.loc[frame.week_start_date.eq("2022-10-14"), column].iloc[0]),
        "oct21": float(frame.loc[frame.week_start_date.eq("2022-10-21"), column].iloc[0]),
        "final_two_week_mean": float(frame.loc[frame.week_start_date.isin(LATE), column].mean()),
    }


def save_audit(output, name, sources, details):
    details = dict(details)
    details["sources"] = {key: {"file": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                          for key, path in sources.items() if path is not None}
    (Path(output) / name).write_text(json.dumps(details, indent=2, allow_nan=False) + "\n")


def plotting():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "hurricane-analysis-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 240, "axes.titlesize": 11})
    return plt


def style_time(ax, ylabel, zero=True):
    import matplotlib.dates as mdates
    ax.axvspan(EVENT_START, EVENT_END, color="#F4ECD3", alpha=.8, zorder=0)
    if zero:
        ax.axhline(0, color="#949DA3", linestyle="--", linewidth=.8, zorder=0)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.FR, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.tick_params(axis="x", rotation=25)
    ax.set_xlabel("Week starting in 2022")
    ax.set_ylabel(ylabel)
    ax.margins(x=.025)


def save_plot(fig, output, name):
    for extension in ("png", "pdf"):
        fig.savefig(Path(output) / f"{name}.{extension}", bbox_inches="tight")
