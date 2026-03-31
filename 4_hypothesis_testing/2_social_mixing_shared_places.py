"""
Hypothesis 2 — Social Mixing at Evacuee-Visited vs Non-Visited Places
======================================================================
Tests whether places visited by evacuees exhibit higher income social mixing
than places that were not visited by evacuees during Hurricane Harvey.

Two analyses:
  1. Snapshot comparison (hurricane week):
       Boxplot of place-level social mixing values and host travel distance,
       split by whether a POI was visited by evacuees.

  2. Weekly trend:
       Average social mixing at shared vs non-shared POIs across 12 weeks,
       tracking whether the gap opens during the hurricane period.

  3. POI taxonomy breakdown:
       Distribution of POI categories for shared vs non-shared places
       visited by the host population during the hurricane week.

Social mixing = 1 − segregation_value (see place_social_mixing.py).

Inputs:
  - evacuees_poi_visits.csv                : evacuee POI visit records
  - host_population_in_top_100_CBG.txt     : host user IDs
  - place_change_rate.txt                  : place-level social mixing × week
  - match_results_remove_home_30/<date>.csv: daily POI visit files
  - GH_poi_list.csv                        : POI metadata
  - taxonomy.csv                           : POI taxonomy mapping
Outputs:
  - results/social_mixing_shared_vs_nonshared.csv
  - results/figures/social_mixing_boxplot.png
  - results/figures/social_mixing_weekly_trend.png
  - results/figures/poi_taxonomy_comparison.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta


# ---------------------------------------------------------------------------
# Configuration — update paths before running
# ---------------------------------------------------------------------------

EVACUEE_VISIT_FILE   = "data/evacuees_poi_visits.csv"
HOST_ID_FILE         = "results/host_population_in_top_100_CBG.txt"
PLACE_CHANGE_FILE    = "data/place_change_rate.txt"
DAILY_VISIT_DIR      = "data/match_results_remove_home_30"
POI_LIST_FILE        = "data/GH_poi_list.csv"
TAXONOMY_FILE        = "data/taxonomy.csv"
OUTPUT_DIR           = "results"
FIGURE_DIR           = os.path.join(OUTPUT_DIR, "figures")

STUDY_START     = pd.Timestamp("2017-07-07")
STUDY_END       = pd.Timestamp("2017-09-28")
EVACUEE_START   = "2017-08-25"
EVACUEE_END     = "2017-09-07"
TARGET_WEEK_DAY = pd.Timestamp("2017-09-08")  # snapshot week
MIN_N_VISITORS  = 4                            # minimum visitors for a place


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_week_ranges(start: pd.Timestamp, end: pd.Timestamp):
    n = ((end - start).days + 1) // 7
    return [(start + timedelta(days=7 * i),
             start + timedelta(days=7 * (i + 1) - 1))
            for i in range(n)]


def load_host_week(week_start, week_end, host_ids: set) -> pd.DataFrame:
    frames = []
    for date in pd.date_range(week_start, week_end).date:
        path = os.path.join(DAILY_VISIT_DIR, f"{date}.csv")
        try:
            df = pd.read_csv(path)
            frames.append(df[df["device_id"].isin(host_ids)])
        except FileNotFoundError:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Load base data
# ---------------------------------------------------------------------------

os.makedirs(FIGURE_DIR, exist_ok=True)

# Evacuee place IDs (disaster window only)
evac_df = pd.read_csv(EVACUEE_VISIT_FILE)
evac_df["date"] = pd.to_datetime(evac_df["date"])
evac_filtered  = evac_df[
    (evac_df["date"] >= EVACUEE_START) &
    (evac_df["date"] <= EVACUEE_END)
]
evac_place_ids = set(evac_filtered["place_id"])

# Host IDs
with open(HOST_ID_FILE) as f:
    host_ids = set(line.strip() for line in f)

# Place-level social mixing data
place_change = pd.read_csv(
    PLACE_CHANGE_FILE,
    names=["place_id", "n", "q1", "q2", "q3", "q4",
           "segregation_value", "week", "change_rate"]
)
# Convert to social mixing (1 − segregation)
place_change["social_mixing_value"] = 1 - place_change["segregation_value"]
place_change["is_shared"] = place_change["place_id"].isin(evac_place_ids)

week_ranges = build_week_ranges(STUDY_START, STUDY_END)


# ---------------------------------------------------------------------------
# Analysis 1 — Snapshot: hurricane week boxplot
# ---------------------------------------------------------------------------

# Identify the week containing the target day
target_week_idx = next(
    i for i, (ws, we) in enumerate(week_ranges) if ws <= TARGET_WEEK_DAY <= we
)
target_ws, target_we = week_ranges[target_week_idx]

seg_week = (
    place_change[place_change["week"] == target_week_idx]
    .query(f"n >= {MIN_N_VISITORS}")
    .copy()
)

# IQR clipping for cleaner visualisation
q1, q3 = seg_week["social_mixing_value"].quantile([0.25, 0.75])
iqr     = q3 - q1
seg_week = seg_week[
    seg_week["social_mixing_value"].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
]
seg_week["is_shared"] = seg_week["is_shared"].map(
    {True: "Visited by evacuees", False: "Not visited"}
)

host_week_df = load_host_week(target_ws, target_we, host_ids)
host_week_df["is_shared"] = host_week_df["place_id"].isin(evac_place_ids)
host_week_df["is_shared"] = host_week_df["is_shared"].map(
    {True: "Visited by evacuees", False: "Not visited"}
)

palette = {"Visited by evacuees": "#1b7f5c", "Not visited": "#b3b3b3"}
order   = ["Visited by evacuees", "Not visited"]
box_kw  = dict(width=0.28, showcaps=True, showfliers=False, saturation=1,
               boxprops=dict(edgecolor="black", linewidth=1.2),
               medianprops=dict(color="black", linewidth=1.4),
               whiskerprops=dict(color="black", linewidth=1.0),
               capprops=dict(color="black", linewidth=1.0))

sns.set(style="white", context="talk")
fig, axes = plt.subplots(1, 2, figsize=(7, 4))

sns.boxplot(data=seg_week, x="is_shared", y="social_mixing_value",
            order=order, palette=palette, ax=axes[0], **box_kw)
axes[0].set_ylabel("Place social mixing", fontsize=13)
axes[0].set_xlabel("")
axes[0].set_ylim(0, 1.0)

sns.boxplot(data=host_week_df, x="is_shared", y="travel_distance",
            order=order, palette=palette, ax=axes[1], **box_kw)
axes[1].set_ylabel("Average travel distance (m)", fontsize=13)
axes[1].set_xlabel("")
axes[1].set_ylim(0, 70_000)

for ax in axes:
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "social_mixing_boxplot.png"), dpi=150)
plt.show()


# ---------------------------------------------------------------------------
# Analysis 2 — Weekly trend: social mixing at shared vs non-shared POIs
# ---------------------------------------------------------------------------

shared_avg     = (place_change[place_change["is_shared"]]
                  .groupby("week")["social_mixing_value"].mean())
non_shared_avg = (place_change[~place_change["is_shared"]]
                  .groupby("week")["social_mixing_value"].mean())

trend_df = pd.DataFrame({
    "week":            shared_avg.index,
    "shared":          shared_avg.values,
    "non_shared":      non_shared_avg.reindex(shared_avg.index).values,
    "week_label":      [
        (STUDY_START + timedelta(days=7 * i)).strftime("%m-%d")
        for i in shared_avg.index
    ]
})
trend_df.to_csv(
    os.path.join(OUTPUT_DIR, "social_mixing_shared_vs_nonshared.csv"),
    index=False
)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(trend_df["week_label"], trend_df["shared"],
        marker="o", color="#1b7f5c", label="Visited by evacuees")
ax.plot(trend_df["week_label"], trend_df["non_shared"],
        marker="s", color="#b3b3b3", label="Not visited")
ax.set_xlabel("Week Starting")
ax.set_ylabel("Avg social mixing value")
ax.set_title("Weekly Social Mixing: Shared vs Non-Shared POIs")
ax.tick_params(axis="x", rotation=45)
ax.legend(frameon=False)
sns.despine(ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "social_mixing_weekly_trend.png"), dpi=150)
plt.show()


# ---------------------------------------------------------------------------
# Analysis 3 — POI taxonomy comparison (hurricane week)
# ---------------------------------------------------------------------------

poi_list  = pd.read_csv(POI_LIST_FILE)
taxo_file = pd.read_csv(TAXONOMY_FILE)
poi_taxo  = (
    poi_list.merge(taxo_file, on=["top_category", "sub_category"], how="left")
            .rename(columns={"safegraph_place_id": "place_id"})
    [["place_id", "Taxonomy"]]
    .dropna(subset=["Taxonomy"])
)

host_disaster = load_host_week(target_ws, target_we, host_ids)
host_place_ids   = set(host_disaster["place_id"])
shared_places    = host_place_ids & evac_place_ids
non_shared_places = host_place_ids - evac_place_ids

shared_taxo = (pd.DataFrame({"place_id": list(shared_places)})
               .merge(poi_taxo, on="place_id", how="left")
               .dropna(subset=["Taxonomy"]))
non_shared_taxo = (pd.DataFrame({"place_id": list(non_shared_places)})
                   .merge(poi_taxo, on="place_id", how="left")
                   .dropna(subset=["Taxonomy"]))

taxo_df = pd.DataFrame({
    "Shared POI":     shared_taxo["Taxonomy"].value_counts(normalize=True),
    "Non-Shared POI": non_shared_taxo["Taxonomy"].value_counts(normalize=True),
}).fillna(0).sort_index().iloc[1:-1]  # drop first/last generic categories

fig, ax = plt.subplots(figsize=(5, 8))
taxo_df.plot(kind="barh", ax=ax, color=["#1b7f5c", "#b0b0b0"],
             width=0.7, edgecolor="none")
ax.set_xlabel("Proportion of POIs", fontsize=14)
ax.set_ylabel("")
ax.tick_params(axis="both", labelsize=13)
ax.legend(["Shared POI", "Non-Shared POI"], frameon=False, fontsize=12,
          loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "poi_taxonomy_comparison.png"), dpi=150)
plt.show()
