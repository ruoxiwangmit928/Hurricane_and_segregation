"""
Host Population Identification and Counterfactual Analysis
==========================================================
Identifies the host population (residents who stayed during Hurricane Harvey)
in the top 100 most-visited CBGs, then constructs counterfactual mobility
baselines at three levels of stratification.

Host population definition:
  Individuals who (1) lived in one of the top 100 evacuee-destination CBGs
  and (2) did NOT evacuate during the hurricane window (Aug 23 – Sep 2,
  displacement >= 4 days).

Counterfactual approach:
  Pre-event POI visit records are randomly subsampled each week so that the
  total dwell time matches the observed host dwell time. This controls for
  the overall activity reduction and isolates composition effects.

  Three levels of stratification:

  Level 1 — Dwell time only:
    Match total weekly dwell time across all visits.
    Output: counterfactual_dwell_time/<week>.csv

  Level 2 — Distance x income quartile:
    Match dwell time separately for each combination of
    travel-distance bin and income quartile (Q1-Q4).
    Output: counterfactual_distance/<week>.csv

  Level 3 — Place category x distance x income quartile:
    Match dwell time separately for each combination of
    POI taxonomy category, travel-distance bin, and income quartile.
    Output: counterfactual_category/<week>.csv

Study period: 2017-07-07 to 2017-09-28 (12 weeks)

Inputs:
  - device_id_cbg_10_days.csv           : home CBG per device
  - displacement_periods.csv            : evacuation episodes
  - displacement_status.csv             : daily displacement status
  - cbg_statistics.csv                  : CBG-level evacuee inflow counts
  - individual_home_income_quantile.csv : income quartile per device
  - host_population_in_top_100_CBG.txt  : host ID list (output of Step 0)
  - match_results_remove_home_30/       : daily POI visit files
  - aggregate_data_before_distance.txt  : pre-event aggregate visit records
  - GH_poi_list.csv + taxonomy.csv      : POI category mapping (Level 3 only)
"""

import os
import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt


# ---------------------------------------------------------------------------
# Configuration — update paths before running
# ---------------------------------------------------------------------------

HOME_CBG_FILE        = "data/device_id_cbg_10_days.csv"
DISPLACEMENT_STATUS  = "results/displacement_status.csv"
DISPLACEMENT_PERIODS = "results/displacement_periods.csv"
CBG_STATS_FILE       = "data/cbg_statistics.csv"
INCOME_QUANTILE_FILE = "data/individual_home_income_quantile.csv"
HOST_ID_FILE         = "results/host_population_in_top_100_CBG.txt"
DAILY_VISIT_DIR      = "data/match_results_remove_home_30"
PRE_EVENT_AGG_FILE   = "data/aggregate_data_before_distance.txt"
POI_LIST_FILE        = "data/GH_poi_list.csv"
TAXONOMY_FILE        = "data/taxonomy.csv"

OUT_DWELL    = "results/counterfactual_dwell_time"
OUT_DISTANCE = "results/counterfactual_distance"
OUT_CATEGORY = "results/counterfactual_category"

TOP_N_CBG           = 100
EVENT_START         = "2017-08-23"
EVENT_END           = "2017-09-02"
MIN_EVACUATION_DAYS = 4
MIN_TRAVEL_DISTANCE = 30
STUDY_START         = "2017-07-07"
N_WEEKS             = 12
RANDOM_SEED         = 10

DISTANCE_BINS = [0, 1000, 3000, 5000, 10_000, 20_000, 40_000, 5_000_000]
INCOME_QUARTS = [1, 2, 3, 4]
TAXONOMIES    = [
    "Transportation", "Entertainment", "Arts / Museum", "Service",
    "Food", "Coffee / Tea", "Health", "Sports", "City / Outdoors",
    "School", "Shopping", "Grocery", "other"
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def haversine(lon1, lat1, lon2, lat2):
    """Return great-circle distance in metres between two WGS-84 points."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6_367_000 * 2 * asin(sqrt(a))


def load_weekly_visits(week_idx, id_set, date_list):
    """Load and concatenate daily POI visit files for one week."""
    days   = date_list[7 * week_idx: 7 * (week_idx + 1)]
    frames = []
    for date in days:
        path = os.path.join(DAILY_VISIT_DIR, f"{date}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        frames.append(df[df["device_id"].isin(id_set)])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def subsample_to_dwell_target(source, target, seed=RANDOM_SEED):
    """
    Randomly drop rows from source until total stay_time approximates target.

    Parameters
    ----------
    source : pd.DataFrame  — pre-event visit records to subsample
    target : float         — target total dwell time (from observed data)
    seed   : int           — random seed for reproducibility

    Returns
    -------
    pd.DataFrame — subsampled records with total stay_time <= target
    """
    if source.empty or target <= 0:
        return source

    np.random.seed(seed)
    subset  = source.copy()
    current = subset["stay_time"].sum()

    # Bulk removal
    n_remove = int(round(len(subset) * (1 - target / current) * 0.95))
    if 0 < n_remove < len(subset):
        subset = subset.drop(np.random.choice(subset.index, n_remove, replace=False))

    # Fine-grained removal
    current = subset["stay_time"].sum()
    while current > target and len(subset) > 0:
        avg    = current / len(subset)
        n_drop = max(1, int((current - target) / avg * 0.25))
        if n_drop == 0:
            break
        subset  = subset.drop(np.random.choice(subset.index,
                                                min(n_drop, len(subset)),
                                                replace=False))
        current = subset["stay_time"].sum()

    return subset


# ---------------------------------------------------------------------------
# Step 0 — Identify host population
# ---------------------------------------------------------------------------

home_df = pd.read_csv(HOME_CBG_FILE)

periods = pd.read_csv(DISPLACEMENT_PERIODS)
periods["start_date"] = pd.to_datetime(periods["start_date"])
periods = periods[
    (periods["start_date"] > pd.to_datetime(EVENT_START)) &
    (periods["start_date"] <= pd.to_datetime(EVENT_END))
]
periods["days_count"] = pd.to_numeric(periods["days_count"], errors="coerce").astype("Int64")
periods     = periods[periods["days_count"] >= MIN_EVACUATION_DAYS]
evacuee_ids = set(periods["device_id"])

status_df = pd.read_csv(DISPLACEMENT_STATUS, usecols=["device_id"])
all_ids   = set(status_df["device_id"].unique())
stay_ids  = all_ids - evacuee_ids

cbg_stats = pd.read_csv(CBG_STATS_FILE)
cbg_stats.rename(columns={"Unnamed: 0": "cbg_id"}, inplace=True)
top_cbg_ids = set(cbg_stats.head(TOP_N_CBG)["cbg_id"])

residents_top = set(home_df[home_df["visitor_home_cbgs"].isin(top_cbg_ids)]["device_id"])
host_ids      = stay_ids & residents_top

print(f"Host population (top {TOP_N_CBG} CBGs, non-evacuees): {len(host_ids):,}")

for d in ["results", OUT_DWELL, OUT_DISTANCE, OUT_CATEGORY]:
    os.makedirs(d, exist_ok=True)

with open(HOST_ID_FILE, "w") as f:
    for uid in host_ids:
        f.write(f"{uid}\n")
print(f"Host IDs saved -> {HOST_ID_FILE}")


# ---------------------------------------------------------------------------
# Shared pre-event dataset
# ---------------------------------------------------------------------------

home_income = pd.read_csv(INCOME_QUANTILE_FILE)
pre_event   = pd.read_csv(PRE_EVENT_AGG_FILE)
pre_event   = pre_event[pre_event["travel_distance"] >= MIN_TRAVEL_DISTANCE]
pre_event   = pre_event[pre_event["device_id"].isin(host_ids)]
pre_event   = pd.merge(pre_event, home_income, on="device_id")

date_list = [d.date() for d in pd.date_range(start=STUDY_START, periods=N_WEEKS * 7)]


# ---------------------------------------------------------------------------
# Level 1 — Dwell-time matched counterfactual (weekly total)
# ---------------------------------------------------------------------------
# Match the total weekly dwell time of the pre-event dataset to the
# observed host dwell time, without any further stratification.

print("\n--- Level 1: dwell-time matching ---")

for week in range(N_WEEKS):
    observed = load_weekly_visits(week, host_ids, date_list)
    if observed.empty:
        continue

    target  = observed["stay_time"].sum()
    matched = subsample_to_dwell_target(pre_event, target)

    if not matched.empty:
        matched.to_csv(os.path.join(OUT_DWELL, f"{week}.csv"), index=False)
        print(f"  Week {week:>2d} | target={target:,.0f}  matched={matched['stay_time'].sum():,.0f}")


# ---------------------------------------------------------------------------
# Level 2 — Distance x income quartile stratified counterfactual
# ---------------------------------------------------------------------------
# Match dwell time separately within each (distance bin, income quartile) cell
# to preserve the observed spatial-reach and income-group composition.

print("\n--- Level 2: distance x income quartile matching ---")

for week in range(N_WEEKS):
    observed = load_weekly_visits(week, host_ids, date_list)
    if observed.empty:
        continue

    for d_lo, d_hi in zip(DISTANCE_BINS[:-1], DISTANCE_BINS[1:]):
        for q in INCOME_QUARTS:

            obs_cell = observed[
                (observed["travel_distance"] >= d_lo) &
                (observed["travel_distance"] <  d_hi) &
                (observed["quantile"] == q)
            ]
            target = obs_cell["stay_time"].sum()
            if target == 0:
                continue

            src_cell = pre_event[
                (pre_event["travel_distance"] >= d_lo) &
                (pre_event["travel_distance"] <  d_hi) &
                (pre_event["quantile"] == q)
            ]
            if src_cell.empty:
                continue

            matched = subsample_to_dwell_target(src_cell, target)
            if not matched.empty:
                matched.to_csv(
                    os.path.join(OUT_DISTANCE, f"{week}.csv"),
                    header=False, index=False, mode="a"
                )

    print(f"  Week {week:>2d} done.")


# ---------------------------------------------------------------------------
# Level 3 — Place category x distance x income quartile stratified
# ---------------------------------------------------------------------------
# Further stratify by POI taxonomy so the counterfactual also controls for
# the category-level visit composition (e.g., grocery vs entertainment).

print("\n--- Level 3: category x distance x income quartile matching ---")

poi_list  = pd.read_csv(POI_LIST_FILE)
taxo_file = pd.read_csv(TAXONOMY_FILE)
poi_taxo  = pd.merge(poi_list, taxo_file,
                     on=["top_category", "sub_category"], how="left")
poi_taxo["Taxonomy"] = poi_taxo["Taxonomy"].fillna("other")
poi_taxo = poi_taxo[["safegraph_place_id", "Taxonomy"]].rename(
    columns={"safegraph_place_id": "place_id"}
)
pre_event_taxo = pd.merge(pre_event, poi_taxo, on="place_id", how="left")

for week in range(N_WEEKS):
    observed = load_weekly_visits(week, host_ids, date_list)
    if observed.empty:
        continue

    obs_taxo = pd.merge(observed, poi_taxo, on="place_id", how="left")

    for cate in TAXONOMIES:
        for d_lo, d_hi in zip(DISTANCE_BINS[:-1], DISTANCE_BINS[1:]):
            for q in INCOME_QUARTS:

                obs_cell = obs_taxo[
                    (obs_taxo["Taxonomy"] == cate) &
                    (obs_taxo["travel_distance"] >= d_lo) &
                    (obs_taxo["travel_distance"] <  d_hi) &
                    (obs_taxo["quantile"] == q)
                ]
                target = obs_cell["stay_time"].sum()
                if target == 0:
                    continue

                src_cell = pre_event_taxo[
                    (pre_event_taxo["Taxonomy"] == cate) &
                    (pre_event_taxo["travel_distance"] >= d_lo) &
                    (pre_event_taxo["travel_distance"] <  d_hi) &
                    (pre_event_taxo["quantile"] == q)
                ]
                # If no pre-event records exist for this cell, use observed directly
                if src_cell.empty:
                    obs_cell.to_csv(
                        os.path.join(OUT_CATEGORY, f"{week}.csv"),
                        header=False, index=False, mode="a"
                    )
                    continue

                matched = subsample_to_dwell_target(src_cell, target)
                if not matched.empty:
                    matched.to_csv(
                        os.path.join(OUT_CATEGORY, f"{week}.csv"),
                        header=False, index=False, mode="a"
                    )

    print(f"  Week {week:>2d} done.")
