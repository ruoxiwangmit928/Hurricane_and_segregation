"""
Exploration Tendency of Host Population — Hurricane Harvey
==========================================================
Computes weekly place-exploration tendency for the host population
residing in the top 100 most-visited CBGs.

Exploration index (per individual per week):
    p = S_t / N
    where S_t = number of unique places visited,
          N   = total number of visits.

Also tracks:
  - Average travel distance per week
  - Total dwell time per week
  - Change relative to a pre-event baseline (weeks 0–5)
  - Correlation between post-event change and CBG median income

Study period: 2017-07-07 to 2017-09-28 (12 weeks, indexed 0–11)
Pre-event baseline  : weeks 0–5
Post-event period   : weeks 8–11

Inputs:
  - mobility_<week>.txt                  : weekly mobility summary per device
  - host_population_in_top_100_CBG.txt   : host user IDs
  - device_id_cbg_10_days.csv            : home CBG per device
  - cbg_statistics.csv                   : CBG-level evacuee inflow
  - geo_change_features.gpkg             : CBG-level demographic features
Outputs:
  - results/exploration_weekly_stats.csv
  - results/exploration_cbg_change.csv
  - results/figures/exploration_trend.png
  - results/figures/exploration_income_scatter.png
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


# ---------------------------------------------------------------------------
# Configuration — update paths before running
# ---------------------------------------------------------------------------

MOBILITY_DIR     = "data/mobility"          # contains mobility_0.txt … mobility_11.txt
HOST_ID_FILE     = "results/host_population_in_top_100_CBG.txt"
HOME_CBG_FILE    = "data/device_id_cbg_10_days.csv"
CBG_STATS_FILE   = "data/cbg_statistics.csv"
GEO_FEATURES     = "data/geo_change_features.gpkg"

OUTPUT_DIR       = "results"
FIGURE_DIR       = os.path.join(OUTPUT_DIR, "figures")

N_WEEKS          = 12
BASELINE_WEEKS   = range(0, 6)    # weeks 0–5
POST_EVENT_WEEKS = range(8, 12)   # weeks 8–11
TOP_N_CBG        = 100
MIN_VISITS       = 5              # minimum visits per user-week to include
MIN_VISITS_DWELL = 20             # stricter threshold for dwell-time analysis


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

os.makedirs(FIGURE_DIR, exist_ok=True)

# Host population IDs
with open(HOST_ID_FILE) as f:
    host_ids = set(line.strip() for line in f)

# Top 100 CBGs
cbg_stats = pd.read_csv(CBG_STATS_FILE)
cbg_stats.rename(columns={"Unnamed: 0": "cbg_id"}, inplace=True)
top_100_cbgs = cbg_stats.head(TOP_N_CBG)

# Home CBG mapping
home_df = pd.read_csv(HOME_CBG_FILE)[["device_id", "visitor_home_cbgs"]]

# Weekly mobility summaries
frames = []
for week in range(N_WEEKS):
    path = os.path.join(MOBILITY_DIR, f"mobility_{week}.txt")
    df   = pd.read_csv(
        path,
        names=["device_id", "rg", "displacement", "distance",
               "place_number", "visits", "dwell_time"]
    )
    df["week"] = week
    frames.append(df)

mobility = pd.concat(frames, ignore_index=True)
mobility_host = mobility[mobility["device_id"].isin(host_ids)].copy()


# ---------------------------------------------------------------------------
# Exploration index
# ---------------------------------------------------------------------------

mobility_host["exploration"] = (
    mobility_host["place_number"] / mobility_host["visits"]
)

# Filter: minimum visit threshold
mobility_filtered = mobility_host[mobility_host["visits"] >= MIN_VISITS]

weekly_exploration = (
    mobility_filtered.groupby("week")["exploration"]
    .agg(["mean", "std"])
    .reset_index()
)

weekly_distance = (
    mobility_filtered.groupby("week")["distance"]
    .agg(["mean", "std"])
    .reset_index()
)

weekly_exploration.to_csv(
    os.path.join(OUTPUT_DIR, "exploration_weekly_stats.csv"), index=False
)


# ---------------------------------------------------------------------------
# CBG-level change: exploration before vs after
# ---------------------------------------------------------------------------

mobility_cbg = pd.merge(mobility_filtered, home_df, on="device_id")
avg_by_cbg   = (
    mobility_cbg.groupby(["week", "visitor_home_cbgs"])["exploration"]
    .mean()
    .reset_index()
)

baseline = (
    avg_by_cbg[avg_by_cbg["week"].isin(BASELINE_WEEKS)]
    .groupby("visitor_home_cbgs")["exploration"].mean()
    .reset_index()
    .rename(columns={"exploration": "exploration_before"})
)
post = (
    avg_by_cbg[avg_by_cbg["week"].isin(POST_EVENT_WEEKS)]
    .groupby("visitor_home_cbgs")["exploration"].mean()
    .reset_index()
    .rename(columns={"exploration": "exploration_after"})
)

cbg_result = baseline.merge(post, on="visitor_home_cbgs", how="left")
cbg_result["change"]      = cbg_result["exploration_before"] - cbg_result["exploration_after"]
cbg_result["change_rate"] = cbg_result["change"] / cbg_result["exploration_before"]


# ---------------------------------------------------------------------------
# CBG-level change: dwell time before vs after
# ---------------------------------------------------------------------------

mobility_dwell = mobility_host[mobility_host["visits"] >= MIN_VISITS_DWELL].copy()
mobility_dwell_cbg = pd.merge(mobility_dwell, home_df, on="device_id")

dwell_by_cbg = (
    mobility_dwell_cbg.groupby(["week", "visitor_home_cbgs"])["dwell_time"]
    .sum().reset_index()
)
dwell_baseline = (
    dwell_by_cbg[dwell_by_cbg["week"].isin(BASELINE_WEEKS)]
    .groupby("visitor_home_cbgs")["dwell_time"].mean().reset_index()
    .rename(columns={"dwell_time": "dwell_time_before"})
)
dwell_post = (
    dwell_by_cbg[dwell_by_cbg["week"].isin(POST_EVENT_WEEKS)]
    .groupby("visitor_home_cbgs")["dwell_time"].mean().reset_index()
    .rename(columns={"dwell_time": "dwell_time_after"})
)
dwell_result = dwell_baseline.merge(dwell_post, on="visitor_home_cbgs", how="left")
dwell_result["dwell_change"]      = dwell_result["dwell_time_before"] - dwell_result["dwell_time_after"]
dwell_result["dwell_change_rate"] = dwell_result["dwell_change"] / dwell_result["dwell_time_before"]


# ---------------------------------------------------------------------------
# Merge with demographic features
# ---------------------------------------------------------------------------

geo   = gpd.read_file(GEO_FEATURES)
geo["visitor_home_cbgs"] = geo["visitor_home_cbgs"].astype("int64")
demo  = geo[["visitor_home_cbgs", "median_income_x", "quantile",
             "pop_White", "pop_Black", "pop_Asian",
             "proverty_rate", "population_density"]]

combined = (
    top_100_cbgs
    .merge(demo,        left_on="cbg_id", right_on="visitor_home_cbgs")
    .merge(cbg_result,  on="visitor_home_cbgs", how="left")
    .merge(dwell_result, on="visitor_home_cbgs", how="left")
)
combined.to_csv(os.path.join(OUTPUT_DIR, "exploration_cbg_change.csv"), index=False)


# ---------------------------------------------------------------------------
# Figure 1 — Weekly exploration trend (with error bars)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(
    weekly_exploration["week"],
    weekly_exploration["mean"],
    yerr=weekly_exploration["std"],
    fmt="-o", capsize=5, ecolor="gray", elinewidth=1.5,
    color="steelblue", markersize=6
)
ax.axvspan(5.5, 7.5, color="gray", alpha=0.2, label="Hurricane period")
ax.set_xlabel("Week (0 = 2017-07-07)")
ax.set_ylabel("Exploration index (mean ± SD)")
ax.set_title("Host population exploration tendency by week")
ax.set_xticks(range(N_WEEKS))
ax.legend(frameon=False)
sns.despine(ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "exploration_trend.png"), dpi=150)
plt.show()


# ---------------------------------------------------------------------------
# Figure 2 — Exploration change vs CBG median income (scatter + regression)
# ---------------------------------------------------------------------------

r, p = stats.pearsonr(
    combined["median_income_x"].dropna(),
    combined["change"].dropna()
)

fig, ax = plt.subplots(figsize=(6, 5))
sns.set_style("white")
sns.regplot(
    x="median_income_x", y="change",
    data=combined,
    scatter_kws={"alpha": 0.6, "color": "gray", "s": 50},
    line_kws={"color": "black", "linewidth": 2},
    ax=ax
)
ax.set_xlabel("CBG median income", fontsize=14)
ax.set_ylabel("Exploration change (before − after)", fontsize=14)
ax.text(0.02, 0.95, f"r = {r:.2f}\np = {p:.3f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "exploration_income_scatter.png"), dpi=150)
plt.show()
