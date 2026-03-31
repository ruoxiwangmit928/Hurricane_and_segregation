"""
Evacuation Identification — Hurricane Harvey (Houston, 2017)
=============================================================
Identifies evacuation episodes from GPS mobility traces using a
rolling-median displacement threshold.

Evacuation criteria:
  - Individual must be at home (minimum distance < DISPLACEMENT_THRESHOLD)
  during the 10-day pre-event window (2017-08-10 to 2017-08-20).
  - Displacement is flagged when minimum distance exceeds the threshold
    for >= MIN_CONSECUTIVE_DAYS consecutive days.
  - Return home is flagged when distance stays below the threshold
    for >= MIN_CONSECUTIVE_DAYS consecutive days.

Inputs:
  - individual_minimum_distance.txt  : daily minimum distance from home per device
  - device_id_cbg_10_days.csv        : device-to-CBG home assignment
  - id_selection.csv                 : active user selection

Outputs:
  - displacement_status.csv          : daily displacement status per device
  - displacement_periods.csv         : evacuation episode start/end/duration
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Configuration — update paths before running
# ---------------------------------------------------------------------------

DISTANCE_FILE     = "data/individual_minimum_distance.txt"
HOME_CBG_FILE     = "data/device_id_cbg_10_days.csv"
ID_SELECTION_FILE = "data/id_selection.csv"

STATUS_OUT_FILE   = "results/displacement_status.csv"
PERIODS_OUT_FILE  = "results/displacement_periods.csv"

DISPLACEMENT_THRESHOLD = 1000   # metres; also tested at 500 m
MIN_CONSECUTIVE_DAYS   = 3      # days required to confirm displacement or return
PRE_EVENT_START        = "2017-08-10"
PRE_EVENT_END          = "2017-08-20"
EVENT_START            = "2017-08-21"
EVENT_END              = "2017-08-31"
MIN_EVACUATION_DAYS    = 4      # minimum episode length to count as evacuation


# ---------------------------------------------------------------------------
# Step 1 — Identify active users present throughout the study period
# ---------------------------------------------------------------------------

home_df   = pd.read_csv(HOME_CBG_FILE)
id_select = pd.read_csv(ID_SELECTION_FILE, names=["device_id", "n", "date"])
id_select = id_select[id_select["device_id"].isin(home_df["device_id"])]

# Keep the top 25% most active users
id_counts = id_select.groupby("device_id")["n"].sum().sort_values(ascending=False)
active_ids = set(id_counts.iloc[: int(len(id_counts) * 0.25)].index)


# ---------------------------------------------------------------------------
# Step 2 — Classify daily displacement status
# ---------------------------------------------------------------------------
# Rule:
#   binary_value = 1 if minimum_distance > DISPLACEMENT_THRESHOLD else 0
#   status = 'displacement' after >= MIN_CONSECUTIVE_DAYS consecutive 1s
#   status = 'home'         after >= MIN_CONSECUTIVE_DAYS consecutive 0s

data = pd.read_csv(
    DISTANCE_FILE,
    names=["date", "device_id", "minimum_distance"]
)
data["date"] = pd.to_datetime(data["date"])

first_write = True

for device_id, group in data.groupby("device_id"):
    individual = group.copy().reset_index(drop=True)
    individual["binary_value"] = (
        individual["minimum_distance"] > DISPLACEMENT_THRESHOLD
    ).astype(int)

    # Only process users who were at home during the pre-event window
    pre_event = individual[
        (individual["date"] >= PRE_EVENT_START) &
        (individual["date"] <= PRE_EVENT_END)
    ]
    if pre_event.empty or pre_event["minimum_distance"].min() >= DISPLACEMENT_THRESHOLD:
        continue

    status_list         = ["home"] * len(individual)
    consecutive_away    = 0
    consecutive_home    = 0

    for idx, row in individual.iterrows():
        if row["binary_value"] == 1:
            consecutive_away += 1
            consecutive_home  = 0
            if consecutive_away >= MIN_CONSECUTIVE_DAYS:
                for i in range(max(0, idx - MIN_CONSECUTIVE_DAYS + 1), len(status_list)):
                    status_list[i] = "displacement"
        else:
            consecutive_home += 1
            consecutive_away  = 0
            if consecutive_home >= MIN_CONSECUTIVE_DAYS:
                for i in range(max(0, idx - MIN_CONSECUTIVE_DAYS + 1), len(status_list)):
                    status_list[i] = "home"

    individual["status"] = status_list
    individual.to_csv(
        STATUS_OUT_FILE,
        index=False,
        header=first_write,
        mode="w" if first_write else "a"
    )
    first_write = False

print(f"Displacement status saved to: {STATUS_OUT_FILE}")


# ---------------------------------------------------------------------------
# Step 3 — Extract evacuation episodes (start, end, duration)
# ---------------------------------------------------------------------------

status_df = pd.read_csv(STATUS_OUT_FILE)
status_df["date"] = pd.to_datetime(status_df["date"])

records = []

for device_id, group in status_df.groupby("device_id"):
    individual   = group.copy()
    period_start = None

    for _, row in individual.iterrows():
        if row["status"] == "displacement" and period_start is None:
            period_start = row["date"]
        elif row["status"] == "home" and period_start is not None:
            days = (row["date"] - period_start).days
            records.append({
                "device_id":  device_id,
                "start_date": period_start,
                "end_date":   row["date"],
                "days_count": days
            })
            period_start = None

    # Close any open episode at the end of the record
    if period_start is not None:
        last_date = individual.iloc[-1]["date"]
        days = (last_date - period_start).days + 1
        records.append({
            "device_id":  device_id,
            "start_date": period_start,
            "end_date":   last_date,
            "days_count": days
        })

periods_df = pd.DataFrame(records)
periods_df.to_csv(PERIODS_OUT_FILE, index=False)
print(f"Displacement periods saved to: {PERIODS_OUT_FILE}")


# ---------------------------------------------------------------------------
# Step 4 — Filter to hurricane-period evacuees and visualise
# ---------------------------------------------------------------------------

periods_df["start_date"] = pd.to_datetime(periods_df["start_date"])
evacuees = periods_df[
    (periods_df["start_date"] > pd.to_datetime(EVENT_START)) &
    (periods_df["start_date"] <= pd.to_datetime(EVENT_END)) &
    (periods_df["days_count"] >= MIN_EVACUATION_DAYS)
]
evacuees["days_count"] = pd.to_numeric(evacuees["days_count"], errors="coerce").astype("Int64")

print(f"\nEvacuees identified : {evacuees['device_id'].nunique():,}")
print(f"Active user pool    : {len(active_ids):,}")
print(f"Evacuation rate     : {evacuees['device_id'].nunique() / len(active_ids):.1%}")

# Distribution of evacuation duration
sns.set(style="white", context="notebook", palette="deep")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.histplot(evacuees["days_count"].dropna(), bins=20, ax=axes[0], color="steelblue")
axes[0].set_xlabel("Evacuation duration (days)")
axes[0].set_ylabel("Number of evacuees")
axes[0].set_title("Evacuation duration distribution")
sns.despine(ax=axes[0])

evac_dates = evacuees["start_date"].sort_values()
sns.histplot(evac_dates, bins=20, ax=axes[1], color="steelblue")
axes[1].set_xlabel("Departure date")
axes[1].set_ylabel("Number of evacuees")
axes[1].set_title("Evacuation departure dates")
axes[1].tick_params(axis="x", rotation=45)
sns.despine(ax=axes[1])

plt.tight_layout()
plt.savefig("results/evacuation_summary.png", dpi=150, bbox_inches="tight")
plt.show()
