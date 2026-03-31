"""
Individual-Level Income Social Mixing During Hurricane Season
=============================================================
Computes weekly individual-level income social mixing indices for active users,
based on the dwell-time-weighted average of place-level income exposure.

Social Mixing Index (individual):
    mixing_i = 1 - (2/3) * sum_q |exposure_q - 0.25|
    where exposure_q = sum_p(w_p * share_qp), weighted by dwell time at each place.
    A value of 1 indicates perfect mixing across income groups; 0 = complete segregation.

Active user criterion: visited places on >= 10 distinct days in every sample month.

Data source : Cuebiq mobility panel (visit-level records)
Study period: 2022-08-19 to 2022-10-30 (Florida hurricane season)
Output      : Weekly CSVs in evac_results/weekly_individual_social_mixing/
"""

import os
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration — update these before running
# ---------------------------------------------------------------------------

# Replace with your own Snowflake table paths
TABLE         = "YOUR_DATABASE.YOUR_SCHEMA.FL_ALL_VISITS_TARGET_PLACES"
START_DATE    = pd.Timestamp("2022-08-19")
END_DATE      = pd.Timestamp("2022-10-30")
PLACE_OUT_DIR = "evac_results/weekly_place_social_mixing"
OUTPUT_DIR    = "evac_results/weekly_individual_social_mixing"

MIN_ACTIVE_DAYS  = 10   # minimum visit days per month to be considered an active user
MIN_PLACE_VISITS = 4    # minimum visitors per place (inherited from place-level filter)
PLACE_CHUNK_SIZE = 500  # batch size for SQL queries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_ymd(dt: pd.Timestamp) -> str:
    return dt.strftime("%Y%m%d")


def compute_individual_social_mixing(group: pd.DataFrame) -> dict:
    """
    Compute individual-level social mixing for a single user.

    Each place visit is weighted by its share of the user's total dwell time.
    Exposure to each income quartile is the weighted average of place-level shares.

    Parameters
    ----------
    group : pd.DataFrame
        Rows for one user; must contain columns:
        stay_time, q1, q2, q3, q4, place_id.

    Returns
    -------
    dict with exposure_q1–q4, social_mixing_value, and N_places.
    """
    total_time = group["stay_time"].sum()
    if total_time == 0:
        return None

    weights = group["stay_time"] / total_time
    exposure = {f"exposure_q{q}": (weights * group[f"q{q}"]).sum() for q in range(1, 5)}

    segregation = (2 / 3.0) * sum(
        abs(exposure[f"exposure_q{q}"] - 0.25) for q in range(1, 5)
    )
    return {
        **{k: round(v, 4) for k, v in exposure.items()},
        "social_mixing_value": round(1 - segregation, 4),
        "N_places": group["place_id"].nunique(),
    }


# ---------------------------------------------------------------------------
# 1. Identify active users (>= MIN_ACTIVE_DAYS visit days per month)
# ---------------------------------------------------------------------------

ACTIVE_USERS_TABLE = "YOUR_DATABASE.YOUR_SCHEMA.FL_ACTIVE_USERS"

snow_engine.read_sql(f"""
    CREATE OR REPLACE TABLE {ACTIVE_USERS_TABLE} AS
    WITH monthly_days AS (
        SELECT
            CUEBIQ_ID,
            DATE_TRUNC('month', TO_DATE(CAST(PROCESSING_DATE AS VARCHAR), 'YYYYMMDD')) AS month,
            COUNT(DISTINCT PROCESSING_DATE) AS visit_days
        FROM {TABLE}
        WHERE PROCESSING_DATE BETWEEN '{to_ymd(START_DATE)}' AND '{to_ymd(END_DATE)}'
        GROUP BY CUEBIQ_ID, month
    )
    SELECT CUEBIQ_ID
    FROM monthly_days
    GROUP BY CUEBIQ_ID
    HAVING MIN(visit_days) >= {MIN_ACTIVE_DAYS}
""")

valid_ids = set(
    snow_engine.read_sql(f"SELECT CUEBIQ_ID FROM {ACTIVE_USERS_TABLE}")
              ["cuebiq_id"].astype(str).tolist()
)
print(f"Active users: {len(valid_ids):,}")


# ---------------------------------------------------------------------------
# 2. Weekly individual social mixing
# ---------------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

for period in pd.period_range(START_DATE, END_DATE - pd.Timedelta(days=1), freq="W-THU"):

    week_start = period.start_time.normalize()
    week_end   = period.end_time.normalize()
    week_index = f"{week_start:%Y%m%d}_{week_end:%Y%m%d}"
    print(f"\nProcessing week: {week_start:%Y-%m-%d} to {week_end:%Y-%m-%d}")

    # ------------------------------------------------------------------
    # Load place-level social mixing for this week
    # ------------------------------------------------------------------
    place_path = os.path.join(PLACE_OUT_DIR, f"place_{week_index}.csv")
    if not os.path.exists(place_path):
        print(f"  Missing place social mixing file — skipping.")
        continue

    place_week = pd.read_csv(place_path).query(f"n >= {MIN_PLACE_VISITS}")

    # ------------------------------------------------------------------
    # Fetch visit records in batches
    # ------------------------------------------------------------------
    place_ids = place_week["place_id"].tolist()
    chunks    = [place_ids[i:i + PLACE_CHUNK_SIZE]
                 for i in range(0, len(place_ids), PLACE_CHUNK_SIZE)]
    all_stays = []

    for chunk_idx, place_chunk in enumerate(chunks):
        print(f"  Fetching batch {chunk_idx + 1}/{len(chunks)}...", end=" ")
        ids_str = ", ".join(str(p) for p in place_chunk)
        chunk = snow_engine.read_sql(f"""
            SELECT CUEBIQ_ID, PLACE_ID, DWELL_TIME_MINUTES AS stay_time
            FROM {TABLE}
            WHERE PROCESSING_DATE >= '{to_ymd(week_start)}'
              AND PROCESSING_DATE <= '{to_ymd(week_end)}'
              AND PLACE_ID IN ({ids_str})
        """)
        all_stays.append(chunk)
        print("done.")

    stay = pd.concat(all_stays, ignore_index=True)
    stay["cuebiq_id"] = stay["cuebiq_id"].astype(str)
    stay = stay[stay["cuebiq_id"].isin(valid_ids)]

    if stay.empty:
        print("  No active user visits found — skipping.")
        del stay, all_stays
        continue

    df = (stay.merge(place_week[["place_id", "q1", "q2", "q3", "q4"]],
                     on="place_id", how="inner")
               .dropna(subset=["q1", "q2", "q3", "q4"]))

    if df.empty:
        print("  No matched records — skipping.")
        del stay, all_stays, df
        continue

    # ------------------------------------------------------------------
    # Compute individual social mixing
    # ------------------------------------------------------------------
    results = []
    groups  = df.groupby("cuebiq_id")

    for i, (user_id, user_df) in enumerate(groups):
        if i % 1000 == 0:
            print(f"  Users processed: {i:,}/{len(groups):,}", end="\r")

        metrics = compute_individual_social_mixing(user_df)
        if metrics is None:
            continue
        results.append({"cuebiq_id": user_id, "week_index": week_index, **metrics})

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        out_path = os.path.join(OUTPUT_DIR, f"indiv_{week_index}.csv")
        res_df.to_csv(out_path, index=False)
        print(f"\n  Saved: {out_path}  ({len(res_df):,} users)")

    del stay, all_stays, df, results, res_df
