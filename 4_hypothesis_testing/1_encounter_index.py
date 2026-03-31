"""
Hypothesis 1 — Evacuee–Host Encounter Index
============================================
Tests whether evacuees and the host population co-visited the same POIs
during the hurricane period, and tracks how this spatial overlap evolves
week by week.

Two complementary metrics are computed per week:

  Encounter Index:
      EI = sum_p(T_hp * T_ep) / (T_h_total * T_e_total)
      where T_hp = total host dwell time at POI p,
            T_ep = total evacuee dwell time at POI p.
      Higher EI indicates more co-presence.

  Host Shared Ratio:
      HSR = sum_{p in shared POIs}(T_hp) / T_h_total
      Fraction of host dwell time spent at evacuee-visited POIs.

Additionally, weekly dwell-time change rates are compared between
evacuee-visited ("shared") and non-visited ("non-shared") POIs.

Study period : 2017-07-07 to 2017-09-28 (12 weeks)
Hurricane landfall: 2017-08-25

Inputs:
  - evacuees_poi_visits.csv              : evacuee POI visit records
  - host_population_in_top_100_CBG.txt   : host user IDs
  - match_results_remove_home_30/<date>.csv : daily POI visit files
Outputs:
  - results/encounter_index_weekly.csv
  - results/figures/encounter_index.png
  - results/figures/shared_stay_change.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Configuration — update paths before running
# ---------------------------------------------------------------------------

EVACUEE_VISIT_FILE = "data/evacuees_poi_visits.csv"
HOST_ID_FILE       = "results/host_population_in_top_100_CBG.txt"
DAILY_VISIT_DIR    = "data/match_results_remove_home_30"
OUTPUT_DIR         = "results"
FIGURE_DIR         = os.path.join(OUTPUT_DIR, "figures")

STUDY_START     = pd.Timestamp("2017-07-07")
STUDY_END       = pd.Timestamp("2017-09-28")
EVACUEE_START   = pd.Timestamp("2017-08-25")
EVACUEE_END     = pd.Timestamp("2017-09-07")
LANDFALL_DATE   = datetime(2017, 8, 25)
BASELINE_WEEKS  = 6          # first N weeks used as pre-event baseline
MIN_DISTANCE    = 50         # metres — sensitivity filter for dwell-time analysis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_week_ranges(start: pd.Timestamp, end: pd.Timestamp):
    n = ((end - start).days + 1) // 7
    return [(start + timedelta(days=7 * i),
             start + timedelta(days=7 * (i + 1) - 1))
            for i in range(n)]


def load_host_week(week_start, week_end, host_ids: set,
                   min_distance: float = 0) -> pd.DataFrame:
    """Load host POI visits for one week, optionally filtering by distance."""
    frames = []
    for date in pd.date_range(week_start, week_end).date:
        path = os.path.join(DAILY_VISIT_DIR, f"{date}.csv")
        try:
            df = pd.read_csv(path)
            df = df[df["device_id"].isin(host_ids)]
            if min_distance > 0:
                df = df[df["travel_distance"] >= min_distance]
            frames.append(df)
        except FileNotFoundError:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Load evacuee data (fixed disaster window)
# ---------------------------------------------------------------------------

os.makedirs(FIGURE_DIR, exist_ok=True)

evac_df = pd.read_csv(EVACUEE_VISIT_FILE)
evac_df["date"] = pd.to_datetime(evac_df["date"])
evac_filtered = evac_df[
    (evac_df["date"] >= EVACUEE_START) &
    (evac_df["date"] <= EVACUEE_END)
]
evac_visits    = (evac_filtered.groupby("place_id")["stay_time"]
                               .sum().reset_index(name="n_evacuee"))
evac_place_ids = set(evac_visits["place_id"])
total_evac_time = evac_visits["n_evacuee"].sum()

# Load host IDs
with open(HOST_ID_FILE) as f:
    host_ids = set(line.strip() for line in f)

week_ranges = build_week_ranges(STUDY_START, STUDY_END)


# ---------------------------------------------------------------------------
# Weekly encounter index and host shared ratio
# ---------------------------------------------------------------------------

results = []

for week_start, week_end in week_ranges:
    df_week = load_host_week(week_start, week_end, host_ids)
    if df_week.empty:
        continue

    host_visits     = (df_week.groupby("place_id")["stay_time"]
                               .sum().reset_index(name="n_host"))
    total_host_time = host_visits["n_host"].sum()
    if total_host_time == 0 or total_evac_time == 0:
        continue

    shared = pd.merge(host_visits, evac_visits, on="place_id", how="inner")
    shared["encounters"] = shared["n_host"] * shared["n_evacuee"]

    encounter_index  = shared["encounters"].sum() / (total_host_time * total_evac_time)
    host_shared_ratio = shared["n_host"].sum() / total_host_time

    host_place_ids   = set(host_visits["place_id"])
    overlap_ratio    = len(host_place_ids & evac_place_ids) / len(host_place_ids)

    results.append({
        "week_start":        week_start.strftime("%Y-%m-%d"),
        "week_label":        week_start.strftime("%m-%d"),
        "encounter_index":   encounter_index,
        "host_shared_ratio": host_shared_ratio,
        "poi_overlap_ratio": overlap_ratio,
    })

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUTPUT_DIR, "encounter_index_weekly.csv"), index=False)


# ---------------------------------------------------------------------------
# Weekly dwell-time change: shared vs non-shared POIs
# ---------------------------------------------------------------------------

shared_times     = []
non_shared_times = []

for week_start, week_end in week_ranges:
    df_week = load_host_week(week_start, week_end, host_ids,
                             min_distance=MIN_DISTANCE)
    if df_week.empty:
        shared_times.append(0)
        non_shared_times.append(0)
        continue

    df_week["is_shared"] = df_week["place_id"].isin(evac_place_ids)
    shared_times.append(df_week[df_week["is_shared"]]["stay_time"].sum())
    non_shared_times.append(df_week[~df_week["is_shared"]]["stay_time"].sum())

shared_baseline     = np.mean(shared_times[:BASELINE_WEEKS])
non_shared_baseline = np.mean(non_shared_times[:BASELINE_WEEKS])
shared_change       = [(v - shared_baseline) / shared_baseline for v in shared_times]
non_shared_change   = [(v - non_shared_baseline) / non_shared_baseline for v in non_shared_times]
week_labels         = [ws.strftime("%m-%d") for ws, _ in week_ranges]


# ---------------------------------------------------------------------------
# Figure 1 — Encounter Index and Host Shared Ratio over time
# ---------------------------------------------------------------------------

week_dts = [datetime.strptime(d, "%m-%d").replace(year=2017)
            for d in results_df["week_label"]]
half_week = timedelta(days=3.5)

fig, ax1 = plt.subplots(figsize=(7, 4))
ax1.plot(week_dts, results_df["encounter_index"],
         color="tab:blue", marker="o", markersize=6, linewidth=1.8,
         markerfacecolor="white", label="Encounter Index")
ax1.set_ylabel("Encounter Index", fontsize=11, color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.plot(week_dts, results_df["host_shared_ratio"],
         color="tab:orange", marker="o", markersize=6, linewidth=1.8,
         markerfacecolor="white", label="Host Shared Ratio")
ax2.set_ylabel("Host Shared Ratio", fontsize=11, color="tab:orange")
ax2.tick_params(axis="y", labelcolor="tab:orange")

ax1.axvspan(LANDFALL_DATE - half_week, LANDFALL_DATE + half_week,
            color="gray", alpha=0.25)
ax1.set_xticks(week_dts)
ax1.set_xticklabels(results_df["week_label"], rotation=45)
ax1.set_xlabel("Week Starting")
fig.suptitle("Weekly Encounter Index and Host Shared Ratio (Harvey 2017)")
fig.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "encounter_index.png"), dpi=150)
plt.show()


# ---------------------------------------------------------------------------
# Figure 2 — Dwell-time change rate: shared vs non-shared POIs
# ---------------------------------------------------------------------------

week_dts_all = [datetime.strptime(d, "%m-%d").replace(year=2017)
                for d in week_labels]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(week_dts_all, shared_change,
        color="tab:green", marker="o", markersize=6, linewidth=1.8,
        markerfacecolor="white", label="Shared POI")
ax.plot(week_dts_all, non_shared_change,
        color="tab:gray", marker="o", markersize=6, linewidth=1.8,
        markerfacecolor="white", label="Non-shared POI")
ax.axvspan(LANDFALL_DATE - half_week, LANDFALL_DATE + half_week,
           color="gray", alpha=0.25)
ax.axhline(0, linestyle="--", color="0.5", linewidth=1)
ax.set_ylabel("Change rate vs pre-disaster baseline", fontsize=11)
ax.set_xlabel("Week Starting")
ax.set_xticks(week_dts_all)
ax.set_xticklabels(week_labels, rotation=45)
ax.legend(frameon=False)
sns.despine(ax=ax)
fig.suptitle("Host Dwell-Time Change: Shared vs Non-Shared POIs")
fig.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "shared_stay_change.png"), dpi=150)
plt.show()
