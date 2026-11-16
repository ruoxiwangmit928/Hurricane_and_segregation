"""Fixed Harvey location reference shared by the main Figure 4 analyses.

The acute reference is Aug 25--Sep 7, 2017, including both dates. Count
distinct identified evacuees, only while within their recorded displacement
episodes. Main overlap curves retain the entire reference; location-group
comparisons use >=3 distinct evacuees. The sets do not change by host week.
"""
from pathlib import Path
import hashlib
import os
import tempfile

import numpy as np
import pandas as pd

REFERENCE_START = pd.Timestamp("2017-08-25")
REFERENCE_END = pd.Timestamp("2017-09-07")
WEEK0 = pd.Timestamp("2017-07-07")
SHARED_THRESHOLD = 3
BASELINE_WEEKS = list(range(6))
GROUPS = ("Shared", "Non-shared")
COLORS = ("#1b7f5c", "#b0b0b0")


def clean_ids(series):
    result = series.astype("string").str.strip()
    return result.mask(result.eq(""))


def checksum(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def build_reference(path):
    """Return complete profile, threshold-three IDs and audit from visit rows."""
    columns = ["device_id", "place_id", "date", "start_date", "end_date", "stay_time"]
    frame = pd.read_csv(path, usecols=columns, dtype="string", low_memory=False)
    for col in ["device_id", "place_id"]:
        frame[col] = clean_ids(frame[col])
    for col in ["date", "start_date", "end_date"]:
        # Source dates are local calendar dates; keep all visits on Sep 7.
        frame[col] = pd.to_datetime(frame[col].str.slice(0, 10),
                                    format="%Y-%m-%d", errors="coerce")
    invalid_dates = int(frame.date.isna().sum())
    acute = frame.loc[frame.date.between(REFERENCE_START, REFERENCE_END)].copy()
    acute["stay_time"] = pd.to_numeric(acute.stay_time, errors="coerce")
    if acute.empty:
        raise ValueError("No visits in the fixed Aug25--Sep7 reference")
    if acute[["device_id", "place_id", "start_date", "end_date"]].isna().any().any() or not np.isfinite(acute.stay_time).all():
        raise ValueError("Invalid IDs, episode dates or dwell in the acute visit records")
    if (acute.start_date > acute.end_date).any():
        raise ValueError("Episode start follows episode end")
    inside = acute.date.between(acute.start_date, acute.end_date, inclusive="both")
    positive = acute.stay_time.ge(0)
    retained = acute.loc[inside & positive]
    profile = retained.groupby("place_id").agg(
        n_evacuees=("device_id", "nunique"), evacuee_dwell=("stay_time", "sum"))
    shared = set(profile.index[profile.n_evacuees.ge(SHARED_THRESHOLD)])
    if profile.empty or not shared or profile.evacuee_dwell.sum() <= 0:
        raise ValueError("Empty reference or no positive dwell; check input coverage")
    audit = dict(reference_start=str(REFERENCE_START.date()),
                 reference_end_inclusive=str(REFERENCE_END.date()),
                 source_rows=len(frame), unparseable_visit_dates_excluded=invalid_dates,
                 acute_period_records_before_episode_filter=len(acute),
                 acute_records_outside_individual_episode=int((~inside).sum()),
                 acute_records_with_negative_dwell=int((~positive).sum()),
                 acute_period_records=len(retained),
                 acute_distinct_evacuees=int(retained.device_id.nunique()),
                 full_reference_locations=len(profile),
                 shared_threshold_distinct_people=SHARED_THRESHOLD,
                 shared_locations=len(shared), shared_set_sha256=checksum(shared),
                 episode_filter="start_date <= visit date <= end_date; finite stay_time >= 0",
                 classification="Fixed once, not recomputed weekly; distinct people, not visit rows")
    return profile, shared, audit


def read_host_activity(path):
    """Prepared table: one row per host-active location/week, not per visit."""
    host = pd.read_csv(path, usecols=["week", "place_id", "host_total_dwell"],
                       dtype={"place_id": "string"})
    host["place_id"] = clean_ids(host.place_id)
    for col in ["week", "host_total_dwell"]:
        host[col] = pd.to_numeric(host[col], errors="raise")
    if host.isna().any().any() or not np.isfinite(host[["week", "host_total_dwell"]]).all().all():
        raise ValueError("Host location-week table contains missing or nonfinite values")
    if host.duplicated(["week", "place_id"]).any():
        raise ValueError("Expected one row per host-active location/week")
    if set(host.week) != set(range(12)) or not host.host_total_dwell.ge(0).all():
        raise ValueError("Expected twelve complete host weeks and nonnegative dwell")
    host["week"] = host.week.astype(int)
    return host


def plotting():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "hurricane-main-figures"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.linewidth": .8, "pdf.fonttype": 42,
                         "svg.fonttype": "none"})
    return plt


def save_plot(fig, output, stem):
    for extension in ("png", "pdf", "svg"):
        fig.savefig(Path(output) / f"{stem}.{extension}", dpi=300,
                    bbox_inches="tight", facecolor="white")
