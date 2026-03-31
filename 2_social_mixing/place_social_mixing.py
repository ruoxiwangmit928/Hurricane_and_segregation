"""
Place-Level Income Social Mixing During Hurricane Season
=========================================================
Computes weekly place-level income social mixing indices from mobility data,
using dwell-time-weighted exposure across income quartiles (Q1–Q4).

Social Mixing Index (adapted from dissimilarity index):
    mixing = 1 - (2/3) * sum_q |share_q - 0.25|
    where share_q = dwell time of quartile q / total dwell time at that place.
    A value of 1 indicates perfect mixing; 0 indicates complete segregation.

Data source : Cuebiq mobility panel (visit-level records)
Study period: 2022-08-05 to 2022-10-30 (Florida hurricane season)
Output      : Weekly CSVs in evac_results/weekly_place_social_mixing/
"""

import os
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration — update these before running
# ---------------------------------------------------------------------------

# Replace with your own Snowflake table path
TABLE      = "YOUR_DATABASE.YOUR_SCHEMA.FL_ALL_VISITS_TARGET_PLACES"
START_DATE = pd.Timestamp("2022-08-05")
END_DATE   = pd.Timestamp("2022-10-30")
OUTPUT_DIR = "evac_results/weekly_place_social_mixing"
INCOME_CSV = "evac_results/fl_cbg_income_quantile.csv"

MIN_VISITORS_PER_PLACE = 4     # minimum unique visitors for a place to be included
PLACE_CHUNK_SIZE       = 2000  # batch size for SQL queries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_ymd(dt: pd.Timestamp) -> str:
    return dt.strftime("%Y%m%d")


def compute_social_mixing(q1: pd.Series, q2: pd.Series,
                          q3: pd.Series, q4: pd.Series) -> pd.Series:
    """
    Compute place-level income social mixing index.

    Social mixing = 1 - dissimilarity, where dissimilarity measures
    deviation of each income group's share from a perfectly mixed baseline (0.25).

    Parameters
    ----------
    q1–q4 : pd.Series
        Share of total dwell time belonging to each income quartile.

    Returns
    -------
    pd.Series of social mixing values in [0, 1], where 1 = perfectly mixed.
    """
    segregation = (2 / 3.0) * (
        (q1 - 0.25).abs() +
        (q2 - 0.25).abs() +
        (q3 - 0.25).abs() +
        (q4 - 0.25).abs()
    )
    return 1 - segregation


# ---------------------------------------------------------------------------
# Load income quartile mapping
# ---------------------------------------------------------------------------

income_df = (
    pd.read_csv(INCOME_CSV, dtype={"cuebiq_id": str})
      .dropna(subset=["income_quartile"])
      .rename(columns={"income_quartile": "quantile"})
)


# ---------------------------------------------------------------------------
# Weekly loop
# ---------------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

for period in pd.period_range(START_DATE, END_DATE - pd.Timedelta(days=1), freq="W-THU"):

    week_start = period.start_time.normalize()
    week_end   = period.end_time.normalize()
    print(f"\nProcessing week: {week_start:%Y-%m-%d} to {week_end:%Y-%m-%d}")

    # ------------------------------------------------------------------
    # 1. Get valid places for this week (minimum visitor threshold)
    # ------------------------------------------------------------------
    place_list = snow_engine.read_sql(f"""
        SELECT PLACE_ID, COUNT(*) AS n
        FROM {TABLE}
        WHERE PROCESSING_DATE >= '{to_ymd(week_start)}'
          AND PROCESSING_DATE <= '{to_ymd(week_end)}'
          AND PLACE_ID IS NOT NULL
        GROUP BY PLACE_ID
        HAVING COUNT(*) >= {MIN_VISITORS_PER_PLACE}
    """)
    place_ids = place_list["place_id"].tolist()
    print(f"  Valid places: {len(place_ids):,}")

    if not place_ids:
        print("  No valid places — skipping.")
        continue

    # ------------------------------------------------------------------
    # 2. Process places in batches and write output
    # ------------------------------------------------------------------
    out_path    = os.path.join(OUTPUT_DIR, f"place_{week_start:%Y%m%d}_{week_end:%Y%m%d}.csv")
    first_chunk = True

    chunks = [place_ids[i:i + PLACE_CHUNK_SIZE]
              for i in range(0, len(place_ids), PLACE_CHUNK_SIZE)]

    for chunk_idx, place_chunk in enumerate(chunks):
        print(f"  Batch {chunk_idx + 1}/{len(chunks)} ({len(place_chunk)} places)...", end=" ")

        ids_str = ", ".join(str(p) for p in place_chunk)
        stay = snow_engine.read_sql(f"""
            SELECT PLACE_ID, CUEBIQ_ID, DWELL_TIME_MINUTES AS stay_time
            FROM {TABLE}
            WHERE PROCESSING_DATE >= '{to_ymd(week_start)}'
              AND PROCESSING_DATE <= '{to_ymd(week_end)}'
              AND PLACE_ID IN ({ids_str})
        """)

        df = (
            stay.assign(cuebiq_id=lambda x: x["cuebiq_id"].astype(str))
                .merge(income_df[["cuebiq_id", "quantile"]], on="cuebiq_id")
                .query("quantile in ['q1','q2','q3','q4']")
        )

        if df.empty:
            print("empty — skipping.")
            del stay, df
            continue

        # Dwell-time sum per place × income quartile
        mat_time = (
            df.groupby(["place_id", "quantile"])["stay_time"].sum()
              .unstack(fill_value=0)
              .reindex(columns=["q1", "q2", "q3", "q4"], fill_value=0)
              .rename(columns={f"q{i}": f"exposure_q{i}" for i in range(1, 5)})
        )

        # Income quartile shares (proportion of total dwell time)
        total_time = mat_time.sum(axis=1)
        shares = (
            mat_time.div(total_time.replace(0, np.nan), axis=0)
                    .rename(columns={f"exposure_q{i}": f"q{i}" for i in range(1, 5)})
        )

        n_visitors = df.groupby("place_id").size()

        out = (
            mat_time.join(shares)
                    .assign(
                        social_mixing_value=compute_social_mixing(
                            shares["q1"], shares["q2"], shares["q3"], shares["q4"]
                        ),
                        n=n_visitors.reindex(mat_time.index).fillna(0).astype(int),
                        week_start=week_start,
                        week_end=week_end,
                    )
                    .reset_index()
            [["week_start", "week_end", "place_id",
              "social_mixing_value", "n", "q1", "q2", "q3", "q4"]]
        )

        out.to_csv(out_path, index=False, mode="w" if first_chunk else "a",
                   header=first_chunk)
        first_chunk = False

        del stay, df, mat_time, total_time, shares, n_visitors, out
        print("done.")

    print(f"  Saved: {out_path}")
