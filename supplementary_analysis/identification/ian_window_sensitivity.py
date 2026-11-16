"""Ian onset-window sensitivity from a standardized daily home trajectory.

Uses 2/3/4 successive observed away records, ±1/2/3-day onset-window shifts,
destination stability and reranked Top-100/Top-200 overlap. The primary
three-observation reconstruction has its own cohort; it is not a substitute
for the main mixing sample.

The trajectory and original-candidate tables must already exist. The engine
must provide execute_statement(sql) and read_sql(sql). Running the analysis
creates or replaces four derived tables under the explicit result prefix.
No credentials or data are stored in this source file.
"""
from pathlib import Path
import argparse
import importlib
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _table_name(value):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*){2}",value):
        raise ValueError("Use an unquoted DATABASE.SCHEMA.TABLE identifier")
    return value.upper()


def run(snow_engine, trajectory_table, original_candidates, result_prefix, output):
    TRAJECTORY_TABLE=_table_name(trajectory_table)
    ORIGINAL_CANDIDATES=_table_name(original_candidates)
    prefix=_table_name(result_prefix)
    CLASSIFICATION_TABLE=prefix+"_CLASSIFICATIONS"
    SPELL_TABLE=prefix+"_SPELLS"
    DESTINATION_TABLE=prefix+"_DESTINATION_METRICS"
    TOP_DEST_TABLE=prefix+"_TOP_DESTINATIONS"
    OUTPUT_DIR=Path(output)
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    outputs={CLASSIFICATION_TABLE,SPELL_TABLE,DESTINATION_TABLE,TOP_DEST_TABLE}
    if outputs & {TRAJECTORY_TABLE,ORIGINAL_CANDIDATES}:
        raise ValueError("Output table names must differ from the input tables")
    TRAJECTORY_START = 20220915
    TRAJECTORY_END = 20221028

    # One-day-at-a-time sensitivity around the main window.
    # Start-date tests hold the end at 2022-10-10.
    # End-date tests hold the start at 2022-09-23.
    WINDOWS = {
        "start_-3d": (20220920, 20221010),
        "start_-2d": (20220921, 20221010),
        "start_-1d": (20220922, 20221010),
        "main":      (20220923, 20221010),
        "start_+1d": (20220924, 20221010),
        "start_+2d": (20220925, 20221010),
        "start_+3d": (20220926, 20221010),
        "end_-3d":   (20220923, 20221007),
        "end_-2d":   (20220923, 20221008),
        "end_-1d":   (20220923, 20221009),
        "end_+1d":   (20220923, 20221011),
        "end_+2d":   (20220923, 20221012),
        "end_+3d":   (20220923, 20221013),
    }
    START_WINDOW_ORDER = [
        "start_-3d", "start_-2d", "start_-1d", "main",
        "start_+1d", "start_+2d", "start_+3d",
    ]
    END_WINDOW_ORDER = [
        "end_-3d", "end_-2d", "end_-1d", "main",
        "end_+1d", "end_+2d", "end_+3d",
    ]
    SHIFT_DAYS = [-3, -2, -1, 0, 1, 2, 3]
    AWAY_THRESHOLDS = (2, 3, 4)

    window_values = ",\n".join(
        f"('{name}', {start}, {end})"
        for name, (start, end) in WINDOWS.items()
    )
    threshold_values = ", ".join(f"({x})" for x in AWAY_THRESHOLDS)


    # ============================================================================
    # Reproduce the original logic:
    # - 2/3/4 away HOME observations;
    # - no observed return home between the first and last away observation;
    # - at least one observed-home record before departure;
    # - keep the earliest validated departure per specification.
    snow_engine.execute_statement(f"""
    CREATE OR REPLACE TABLE {CLASSIFICATION_TABLE} AS
    WITH windows AS (
        SELECT
            COLUMN1::VARCHAR AS WINDOW_NAME,
            COLUMN2::NUMBER AS WINDOW_START,
            COLUMN3::NUMBER AS WINDOW_END
        FROM VALUES
            {window_values}
    ),
    thresholds AS (
        SELECT COLUMN1::NUMBER AS MIN_AWAY_OBS
        FROM VALUES {threshold_values}
    ),
    ordered AS (
        SELECT
            T.*,
            ROW_NUMBER() OVER (
                PARTITION BY CUEBIQ_ID ORDER BY SNAPSHOT_DATE
            ) AS OBS_SEQ,
            SUM(IFF(AWAY_FROM_PRE_HOME = 0, 1, 0)) OVER (
                PARTITION BY CUEBIQ_ID
                ORDER BY SNAPSHOT_DATE
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS HOME_BREAK_GROUP
        FROM {TRAJECTORY_TABLE} T
    ),
    away_runs AS (
        SELECT
            CUEBIQ_ID,
            PRE_EVENT_HOME_CBG,
            HOME_BREAK_GROUP,
            MIN(SNAPSHOT_EVENT_DATE) AS RUN_START_DATE,
            MAX(SNAPSHOT_EVENT_DATE) AS RUN_END_DATE,
            MIN(SNAPSHOT_DATE) AS RUN_START_D,
            MAX(SNAPSHOT_DATE) AS RUN_END_D,
            COUNT(*) AS N_AWAY_OBS,
            DATEDIFF('day', MIN(SNAPSHOT_DATE), MAX(SNAPSHOT_DATE))
                AS RUN_CALENDAR_SPAN_DAYS
        FROM ordered
        WHERE AWAY_FROM_PRE_HOME = 1
        GROUP BY CUEBIQ_ID, PRE_EVENT_HOME_CBG, HOME_BREAK_GROUP
    ),
    eligible AS (
        SELECT
            W.WINDOW_NAME,
            W.WINDOW_START,
            W.WINDOW_END,
            H.MIN_AWAY_OBS,
            R.*
        FROM away_runs R
        CROSS JOIN windows W
        CROSS JOIN thresholds H
        WHERE R.N_AWAY_OBS >= H.MIN_AWAY_OBS
          AND R.RUN_START_DATE BETWEEN W.WINDOW_START AND W.WINDOW_END
          AND EXISTS (
              SELECT 1
              FROM ordered P
              WHERE P.CUEBIQ_ID = R.CUEBIQ_ID
                AND P.SNAPSHOT_DATE < R.RUN_START_D
                AND P.AWAY_FROM_PRE_HOME = 0
          )
    ),
    ranked AS (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY WINDOW_NAME, MIN_AWAY_OBS, CUEBIQ_ID
                ORDER BY RUN_START_D
            ) AS RN
        FROM eligible
    )
    SELECT
        WINDOW_NAME,
        WINDOW_START,
        WINDOW_END,
        MIN_AWAY_OBS,
        CUEBIQ_ID,
        PRE_EVENT_HOME_CBG,
        RUN_START_DATE AS FIRST_AWAY_DATE,
        RUN_END_DATE AS QUALIFYING_RUN_END_DATE,
        N_AWAY_OBS AS QUALIFYING_RUN_AWAY_OBS,
        RUN_CALENDAR_SPAN_DAYS
    FROM ranked
    WHERE RN = 1
    """)

    classification_summary = snow_engine.read_sql(f"""
    SELECT
        WINDOW_NAME,
        MIN_AWAY_OBS,
        COUNT(DISTINCT CUEBIQ_ID) AS N_EVACUEES,
        MIN(FIRST_AWAY_DATE) AS FIRST_DEPARTURE,
        MAX(FIRST_AWAY_DATE) AS LAST_DEPARTURE,
        MEDIAN(QUALIFYING_RUN_AWAY_OBS) AS MEDIAN_AWAY_OBS,
        MEDIAN(RUN_CALENDAR_SPAN_DAYS) AS MEDIAN_RUN_SPAN_DAYS
    FROM {CLASSIFICATION_TABLE}
    GROUP BY WINDOW_NAME, MIN_AWAY_OBS
    ORDER BY WINDOW_NAME, MIN_AWAY_OBS
    """)
    classification_summary.columns = classification_summary.columns.str.lower()
    print(classification_summary)
    classification_summary.to_csv(
        OUTPUT_DIR / "table_window_threshold_sample_sizes.csv", index=False
    )


    # ============================================================================
    overlap = snow_engine.read_sql(f"""
    WITH specs AS (
        SELECT DISTINCT WINDOW_NAME, MIN_AWAY_OBS
        FROM {CLASSIFICATION_TABLE}
    ),
    primary_main_ids AS (
        SELECT DISTINCT CUEBIQ_ID
        FROM {CLASSIFICATION_TABLE}
        WHERE WINDOW_NAME = 'main' AND MIN_AWAY_OBS = 3
    ),
    same_threshold_main AS (
        SELECT DISTINCT MIN_AWAY_OBS, CUEBIQ_ID
        FROM {CLASSIFICATION_TABLE}
        WHERE WINDOW_NAME = 'main'
    ),
    spec_counts AS (
        SELECT
            WINDOW_NAME,
            MIN_AWAY_OBS,
            COUNT(DISTINCT CUEBIQ_ID) AS N_SPEC
        FROM {CLASSIFICATION_TABLE}
        GROUP BY WINDOW_NAME, MIN_AWAY_OBS
    ),
    primary_intersections AS (
        SELECT
            C.WINDOW_NAME,
            C.MIN_AWAY_OBS,
            COUNT(DISTINCT C.CUEBIQ_ID) AS N_INTERSECTION
        FROM {CLASSIFICATION_TABLE} C
        JOIN primary_main_ids M
          ON C.CUEBIQ_ID = M.CUEBIQ_ID
        GROUP BY C.WINDOW_NAME, C.MIN_AWAY_OBS
    ),
    same_threshold_intersections AS (
        SELECT
            C.WINDOW_NAME,
            C.MIN_AWAY_OBS,
            COUNT(DISTINCT C.CUEBIQ_ID) AS N_INTERSECTION_SAME_THRESHOLD
        FROM {CLASSIFICATION_TABLE} C
        JOIN same_threshold_main M
          ON C.MIN_AWAY_OBS = M.MIN_AWAY_OBS
         AND C.CUEBIQ_ID = M.CUEBIQ_ID
        GROUP BY C.WINDOW_NAME, C.MIN_AWAY_OBS
    ),
    main_counts AS (
        SELECT MIN_AWAY_OBS, COUNT(*) AS N_MAIN_SAME_THRESHOLD
        FROM same_threshold_main
        GROUP BY MIN_AWAY_OBS
    )
    SELECT
        S.WINDOW_NAME,
        S.MIN_AWAY_OBS,
        S.N_SPEC,
        (SELECT COUNT(*) FROM primary_main_ids) AS N_PRIMARY_MAIN,
        P.N_INTERSECTION AS N_INTERSECTION_PRIMARY_MAIN,
        P.N_INTERSECTION / NULLIF(
            S.N_SPEC + (SELECT COUNT(*) FROM primary_main_ids)
            - P.N_INTERSECTION, 0
        ) AS JACCARD_WITH_PRIMARY_MAIN,
        M.N_MAIN_SAME_THRESHOLD,
        I.N_INTERSECTION_SAME_THRESHOLD,
        I.N_INTERSECTION_SAME_THRESHOLD / NULLIF(
            S.N_SPEC + M.N_MAIN_SAME_THRESHOLD
            - I.N_INTERSECTION_SAME_THRESHOLD, 0
        ) AS JACCARD_WITH_MAIN_SAME_THRESHOLD
    FROM spec_counts S
    JOIN primary_intersections P
      ON S.WINDOW_NAME = P.WINDOW_NAME
     AND S.MIN_AWAY_OBS = P.MIN_AWAY_OBS
    JOIN same_threshold_intersections I
      ON S.WINDOW_NAME = I.WINDOW_NAME
     AND S.MIN_AWAY_OBS = I.MIN_AWAY_OBS
    JOIN main_counts M
      ON S.MIN_AWAY_OBS = M.MIN_AWAY_OBS
    ORDER BY S.WINDOW_NAME, S.MIN_AWAY_OBS
    """)
    overlap.columns = overlap.columns.str.lower()
    print(overlap)
    overlap.to_csv(OUTPUT_DIR / "table_overlap_with_main.csv", index=False)

    original_overlap = snow_engine.read_sql(f"""
    WITH new_main AS (
        SELECT DISTINCT CUEBIQ_ID
        FROM {CLASSIFICATION_TABLE}
        WHERE WINDOW_NAME = 'main' AND MIN_AWAY_OBS = 3
    ),
    old_main AS (
        SELECT DISTINCT CUEBIQ_ID
        FROM {ORIGINAL_CANDIDATES}
    )
    SELECT
        (SELECT COUNT(*) FROM old_main) AS N_ORIGINAL,
        (SELECT COUNT(*) FROM new_main) AS N_REBUILT_MAIN,
        COUNT(*) AS N_INTERSECTION,
        COUNT(*) / NULLIF(
            (SELECT COUNT(*) FROM old_main)
            + (SELECT COUNT(*) FROM new_main)
            - COUNT(*), 0
        ) AS JACCARD
    FROM old_main O
    JOIN new_main N
      ON O.CUEBIQ_ID = N.CUEBIQ_ID
    """)
    original_overlap.columns = original_overlap.columns.str.lower()
    print(original_overlap)
    original_overlap.to_csv(
        OUTPUT_DIR / "table_original_vs_rebuilt_main.csv", index=False
    )


    # ============================================================================
    snow_engine.execute_statement(f"""
    CREATE OR REPLACE TABLE {SPELL_TABLE} AS
    WITH returns AS (
        SELECT
            C.WINDOW_NAME,
            C.MIN_AWAY_OBS,
            C.CUEBIQ_ID,
            MIN(T.SNAPSHOT_EVENT_DATE) AS RETURN_DATE
        FROM {CLASSIFICATION_TABLE} C
        JOIN {TRAJECTORY_TABLE} T
          ON C.CUEBIQ_ID = T.CUEBIQ_ID
         AND T.SNAPSHOT_EVENT_DATE > C.FIRST_AWAY_DATE
         AND T.AWAY_FROM_PRE_HOME = 0
        GROUP BY C.WINDOW_NAME, C.MIN_AWAY_OBS, C.CUEBIQ_ID
    ),
    last_obs AS (
        SELECT CUEBIQ_ID, MAX(SNAPSHOT_EVENT_DATE) AS LAST_OBSERVED_DATE
        FROM {TRAJECTORY_TABLE}
        GROUP BY CUEBIQ_ID
    )
    SELECT
        C.*,
        R.RETURN_DATE,
        L.LAST_OBSERVED_DATE,
        COALESCE(R.RETURN_DATE, L.LAST_OBSERVED_DATE) AS EFFECTIVE_END_DATE,
        DATEDIFF(
            'day',
            TO_DATE(C.FIRST_AWAY_DATE::STRING, 'YYYYMMDD'),
            TO_DATE(
                COALESCE(R.RETURN_DATE, L.LAST_OBSERVED_DATE)::STRING,
                'YYYYMMDD'
            )
        ) AS DISPLACEMENT_DAYS,
        IFF(R.RETURN_DATE IS NULL, 1, 0) AS IS_RIGHT_CENSORED
    FROM {CLASSIFICATION_TABLE} C
    LEFT JOIN returns R
      ON C.WINDOW_NAME = R.WINDOW_NAME
     AND C.MIN_AWAY_OBS = R.MIN_AWAY_OBS
     AND C.CUEBIQ_ID = R.CUEBIQ_ID
    JOIN last_obs L
      ON C.CUEBIQ_ID = L.CUEBIQ_ID
    """)

    spell_summary = snow_engine.read_sql(f"""
    SELECT
        WINDOW_NAME,
        MIN_AWAY_OBS,
        COUNT(*) AS N_EVACUEES,
        AVG(IFF(RETURN_DATE IS NOT NULL, 1, 0)) AS OBSERVED_RETURN_RATE,
        AVG(IS_RIGHT_CENSORED) AS RIGHT_CENSORED_SHARE,
        MEDIAN(DISPLACEMENT_DAYS) AS MEDIAN_DISPLACEMENT_DAYS,
        AVG(DISPLACEMENT_DAYS) AS MEAN_DISPLACEMENT_DAYS
    FROM {SPELL_TABLE}
    GROUP BY WINDOW_NAME, MIN_AWAY_OBS
    ORDER BY WINDOW_NAME, MIN_AWAY_OBS
    """)
    spell_summary.columns = spell_summary.columns.str.lower()
    print(spell_summary)
    spell_summary.to_csv(OUTPUT_DIR / "table_spell_sensitivity.csv", index=False)


    # ============================================================================
    # Calculate person-level destination metrics from all away HOME observations
    # during each displacement spell.
    snow_engine.execute_statement(f"""
    CREATE OR REPLACE TABLE {DESTINATION_TABLE} AS
    WITH away_sequence AS (
        SELECT
            S.WINDOW_NAME,
            S.MIN_AWAY_OBS,
            S.CUEBIQ_ID,
            S.PRE_EVENT_HOME_CBG,
            S.FIRST_AWAY_DATE,
            S.RETURN_DATE,
            S.EFFECTIVE_END_DATE,
            S.DISPLACEMENT_DAYS,
            S.IS_RIGHT_CENSORED,
            T.SNAPSHOT_EVENT_DATE,
            T.SNAPSHOT_DATE,
            T.CURRENT_HOME_CBG,
            T.CONFIDENCE_LEVEL,
            LAG(T.CURRENT_HOME_CBG) OVER (
                PARTITION BY S.WINDOW_NAME, S.MIN_AWAY_OBS, S.CUEBIQ_ID
                ORDER BY T.SNAPSHOT_DATE
            ) AS PREVIOUS_DESTINATION,
            LAG(T.SNAPSHOT_DATE) OVER (
                PARTITION BY S.WINDOW_NAME, S.MIN_AWAY_OBS, S.CUEBIQ_ID
                ORDER BY T.SNAPSHOT_DATE
            ) AS PREVIOUS_OBS_DATE
        FROM {SPELL_TABLE} S
        JOIN {TRAJECTORY_TABLE} T
          ON S.CUEBIQ_ID = T.CUEBIQ_ID
         AND T.SNAPSHOT_EVENT_DATE >= S.FIRST_AWAY_DATE
         AND T.SNAPSHOT_EVENT_DATE <= S.EFFECTIVE_END_DATE
         AND T.AWAY_FROM_PRE_HOME = 1
    ),
    destination_counts AS (
        SELECT
            WINDOW_NAME,
            MIN_AWAY_OBS,
            CUEBIQ_ID,
            CURRENT_HOME_CBG,
            COUNT(*) AS N_DESTINATION_OBS,
            AVG(CONFIDENCE_LEVEL) AS AVG_DEST_CONFIDENCE,
            MIN(SNAPSHOT_EVENT_DATE) AS FIRST_OBSERVED_AT_DEST,
            MAX(SNAPSHOT_EVENT_DATE) AS LAST_OBSERVED_AT_DEST
        FROM away_sequence
        GROUP BY WINDOW_NAME, MIN_AWAY_OBS, CUEBIQ_ID, CURRENT_HOME_CBG
    ),
    primary_destination AS (
        SELECT *
        FROM destination_counts
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY WINDOW_NAME, MIN_AWAY_OBS, CUEBIQ_ID
            ORDER BY
                N_DESTINATION_OBS DESC,
                AVG_DEST_CONFIDENCE DESC,
                FIRST_OBSERVED_AT_DEST,
                CURRENT_HOME_CBG
        ) = 1
    ),
    person_sequence AS (
        SELECT
            WINDOW_NAME,
            MIN_AWAY_OBS,
            CUEBIQ_ID,
            COUNT(*) AS N_OBSERVED_DESTINATION_DAYS,
            COUNT(DISTINCT CURRENT_HOME_CBG) AS N_UNIQUE_DESTINATION_CBGS,
            COUNT_IF(
                PREVIOUS_DESTINATION IS NOT NULL
                AND CURRENT_HOME_CBG <> PREVIOUS_DESTINATION
            ) AS N_DESTINATION_SWITCHES,
            AVG(IFF(
                PREVIOUS_DESTINATION IS NOT NULL
                AND CURRENT_HOME_CBG <> PREVIOUS_DESTINATION,
                1, 0
            )) AS SWITCH_RATE,
            MEDIAN(DATEDIFF('day', PREVIOUS_OBS_DATE, SNAPSHOT_DATE))
                AS MEDIAN_OBSERVATION_GAP_DAYS,
            MAX(DATEDIFF('day', PREVIOUS_OBS_DATE, SNAPSHOT_DATE))
                AS MAX_OBSERVATION_GAP_DAYS
        FROM away_sequence
        GROUP BY WINDOW_NAME, MIN_AWAY_OBS, CUEBIQ_ID
    ),
    entropy_terms AS (
        SELECT
            D.WINDOW_NAME,
            D.MIN_AWAY_OBS,
            D.CUEBIQ_ID,
            -SUM(
                (D.N_DESTINATION_OBS / NULLIF(P.N_OBSERVED_DESTINATION_DAYS, 0))
                * LN(D.N_DESTINATION_OBS / NULLIF(P.N_OBSERVED_DESTINATION_DAYS, 0))
            ) AS DESTINATION_ENTROPY
        FROM destination_counts D
        JOIN person_sequence P
          ON D.WINDOW_NAME = P.WINDOW_NAME
         AND D.MIN_AWAY_OBS = P.MIN_AWAY_OBS
         AND D.CUEBIQ_ID = P.CUEBIQ_ID
        GROUP BY D.WINDOW_NAME, D.MIN_AWAY_OBS, D.CUEBIQ_ID
    )
    SELECT
        S.*,
        P.CURRENT_HOME_CBG AS PRIMARY_DESTINATION_CBG,
        P.N_DESTINATION_OBS AS PRIMARY_DESTINATION_DAYS,
        P.N_DESTINATION_OBS / NULLIF(Q.N_OBSERVED_DESTINATION_DAYS, 0)
            AS PRIMARY_DESTINATION_SHARE,
        P.AVG_DEST_CONFIDENCE,
        P.FIRST_OBSERVED_AT_DEST,
        P.LAST_OBSERVED_AT_DEST,
        Q.N_OBSERVED_DESTINATION_DAYS,
        Q.N_UNIQUE_DESTINATION_CBGS,
        Q.N_DESTINATION_SWITCHES,
        Q.SWITCH_RATE,
        Q.MEDIAN_OBSERVATION_GAP_DAYS,
        Q.MAX_OBSERVATION_GAP_DAYS,
        E.DESTINATION_ENTROPY,
        CASE
            WHEN Q.N_OBSERVED_DESTINATION_DAYS < 3
                THEN 'Insufficient observations'
            WHEN P.N_DESTINATION_OBS / NULLIF(Q.N_OBSERVED_DESTINATION_DAYS, 0)
                    >= 0.80
                 AND Q.N_DESTINATION_SWITCHES <= 1
                THEN 'Stable primary destination'
            WHEN P.N_DESTINATION_OBS / NULLIF(Q.N_OBSERVED_DESTINATION_DAYS, 0)
                    >= 0.60
                THEN 'Mostly primary with excursions'
            WHEN Q.N_DESTINATION_SWITCHES >= 3
                THEN 'Cycling or multi-stop'
            ELSE 'Sequential multi-destination'
        END AS TRAJECTORY_TYPE
    FROM {SPELL_TABLE} S
    LEFT JOIN primary_destination P
      ON S.WINDOW_NAME = P.WINDOW_NAME
     AND S.MIN_AWAY_OBS = P.MIN_AWAY_OBS
     AND S.CUEBIQ_ID = P.CUEBIQ_ID
    LEFT JOIN person_sequence Q
      ON S.WINDOW_NAME = Q.WINDOW_NAME
     AND S.MIN_AWAY_OBS = Q.MIN_AWAY_OBS
     AND S.CUEBIQ_ID = Q.CUEBIQ_ID
    LEFT JOIN entropy_terms E
      ON S.WINDOW_NAME = E.WINDOW_NAME
     AND S.MIN_AWAY_OBS = E.MIN_AWAY_OBS
     AND S.CUEBIQ_ID = E.CUEBIQ_ID
    """)

    destination_summary = snow_engine.read_sql(f"""
    SELECT
        WINDOW_NAME,
        MIN_AWAY_OBS,
        TRAJECTORY_TYPE,
        COUNT(*) AS N_EVACUEES,
        COUNT(*) / SUM(COUNT(*)) OVER (
            PARTITION BY WINDOW_NAME, MIN_AWAY_OBS
        ) AS SHARE_EVACUEES,
        MEDIAN(PRIMARY_DESTINATION_SHARE) AS MEDIAN_PRIMARY_SHARE,
        MEDIAN(N_UNIQUE_DESTINATION_CBGS) AS MEDIAN_UNIQUE_DESTINATIONS,
        MEDIAN(N_DESTINATION_SWITCHES) AS MEDIAN_SWITCHES,
        MEDIAN(DISPLACEMENT_DAYS) AS MEDIAN_DISPLACEMENT_DAYS,
        AVG(IS_RIGHT_CENSORED) AS RIGHT_CENSORED_SHARE
    FROM {DESTINATION_TABLE}
    GROUP BY WINDOW_NAME, MIN_AWAY_OBS, TRAJECTORY_TYPE
    ORDER BY WINDOW_NAME, MIN_AWAY_OBS, N_EVACUEES DESC
    """)
    destination_summary.columns = destination_summary.columns.str.lower()
    print(destination_summary)
    destination_summary.to_csv(
        OUTPUT_DIR / "table_destination_trajectory_types.csv", index=False
    )


    # ============================================================================
    snow_engine.execute_statement(f"""
    CREATE OR REPLACE TABLE {TOP_DEST_TABLE} AS
    WITH counts AS (
        SELECT
            WINDOW_NAME,
            MIN_AWAY_OBS,
            PRIMARY_DESTINATION_CBG,
            COUNT(DISTINCT CUEBIQ_ID) AS N_RECEIVED
        FROM {DESTINATION_TABLE}
        WHERE PRIMARY_DESTINATION_CBG IS NOT NULL
        GROUP BY WINDOW_NAME, MIN_AWAY_OBS, PRIMARY_DESTINATION_CBG
    )
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY WINDOW_NAME, MIN_AWAY_OBS
            ORDER BY N_RECEIVED DESC, PRIMARY_DESTINATION_CBG
        ) AS DESTINATION_RANK
    FROM counts
    """)

    top_destination_overlap = snow_engine.read_sql(f"""
    WITH top_sizes AS (
        SELECT COLUMN1::NUMBER AS TOP_N
        FROM VALUES (100), (200)
    ),
    specs AS (
        SELECT DISTINCT WINDOW_NAME, MIN_AWAY_OBS
        FROM {TOP_DEST_TABLE}
    ),
    main_sets AS (
        SELECT
            T.MIN_AWAY_OBS,
            N.TOP_N,
            T.PRIMARY_DESTINATION_CBG
        FROM {TOP_DEST_TABLE} T
        CROSS JOIN top_sizes N
        WHERE T.WINDOW_NAME = 'main'
          AND T.DESTINATION_RANK <= N.TOP_N
    ),
    spec_sets AS (
        SELECT
            T.WINDOW_NAME,
            T.MIN_AWAY_OBS,
            N.TOP_N,
            T.PRIMARY_DESTINATION_CBG
        FROM {TOP_DEST_TABLE} T
        CROSS JOIN top_sizes N
        WHERE T.DESTINATION_RANK <= N.TOP_N
    ),
    set_sizes AS (
        SELECT
            WINDOW_NAME,
            MIN_AWAY_OBS,
            TOP_N,
            COUNT(*) AS N_SPEC_DESTINATIONS
        FROM spec_sets
        GROUP BY WINDOW_NAME, MIN_AWAY_OBS, TOP_N
    ),
    intersections AS (
        SELECT
            S.WINDOW_NAME,
            S.MIN_AWAY_OBS,
            S.TOP_N,
            COUNT(*) AS N_OVERLAP_MAIN
        FROM spec_sets S
        JOIN main_sets M
          ON S.MIN_AWAY_OBS = M.MIN_AWAY_OBS
         AND S.TOP_N = M.TOP_N
         AND S.PRIMARY_DESTINATION_CBG = M.PRIMARY_DESTINATION_CBG
        GROUP BY S.WINDOW_NAME, S.MIN_AWAY_OBS, S.TOP_N
    )
    SELECT
        Z.WINDOW_NAME,
        Z.MIN_AWAY_OBS,
        Z.TOP_N,
        Z.N_SPEC_DESTINATIONS,
        I.N_OVERLAP_MAIN,
        I.N_OVERLAP_MAIN / NULLIF(
            Z.N_SPEC_DESTINATIONS + Z.TOP_N - I.N_OVERLAP_MAIN,
            0
        ) AS JACCARD_WITH_MAIN_DESTINATIONS
    FROM set_sizes Z
    JOIN intersections I
      ON Z.WINDOW_NAME = I.WINDOW_NAME
     AND Z.MIN_AWAY_OBS = I.MIN_AWAY_OBS
     AND Z.TOP_N = I.TOP_N
    ORDER BY Z.TOP_N, Z.WINDOW_NAME, Z.MIN_AWAY_OBS
    """)
    top_destination_overlap.columns = top_destination_overlap.columns.str.lower()
    print(top_destination_overlap)
    top_destination_overlap.to_csv(
        OUTPUT_DIR / "table_top_destination_overlap.csv",
        index=False,
    )


    # ============================================================================
    # Final reviewer-facing plots: evacuee overlap and Top-100/200 CBG overlap
    # ============================================================================
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex="col")

    for threshold in AWAY_THRESHOLDS:
        label = f">={threshold} away observations"
        for col, order in enumerate([START_WINDOW_ORDER, END_WINDOW_ORDER]):
            counts_panel = (
                classification_summary[
                    classification_summary["min_away_obs"] == threshold
                ]
                .set_index("window_name")
                .reindex(order)
                .reset_index()
            )
            overlap_panel = (
                overlap[overlap["min_away_obs"] == threshold]
                .set_index("window_name")
                .reindex(order)
                .reset_index()
            )
            axes[0, col].plot(
                SHIFT_DAYS,
                counts_panel["n_evacuees"],
                marker="o",
                label=label,
            )
            axes[1, col].plot(
                SHIFT_DAYS,
                overlap_panel["jaccard_with_main_same_threshold"],
                marker="o",
                label=label,
            )

    axes[0, 0].set_title("Move start date; end fixed")
    axes[0, 1].set_title("Move end date; start fixed")
    axes[0, 0].set_ylabel("Evacuee sample size")
    axes[1, 0].set_ylabel("Evacuee Jaccard with main")
    axes[1, 0].set_xlabel("Date shift from main (days)")
    axes[1, 1].set_xlabel("Date shift from main (days)")
    for ax in axes.flat:
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(SHIFT_DAYS)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "figure_evacuee_daily_window_overlap.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex="col", sharey=True)
    for row, top_n in enumerate([100, 200]):
        top_panel = top_destination_overlap[
            top_destination_overlap["top_n"] == top_n
        ]
        for threshold in AWAY_THRESHOLDS:
            label = f">={threshold} away observations"
            for col, order in enumerate([START_WINDOW_ORDER, END_WINDOW_ORDER]):
                panel = (
                    top_panel[top_panel["min_away_obs"] == threshold]
                    .set_index("window_name")
                    .reindex(order)
                    .reset_index()
                )
                axes[row, col].plot(
                    SHIFT_DAYS,
                    panel["jaccard_with_main_destinations"],
                    marker="o",
                    label=label,
                )

    axes[0, 0].set_title("Top 100: move start date")
    axes[0, 1].set_title("Top 100: move end date")
    axes[1, 0].set_title("Top 200: move start date")
    axes[1, 1].set_title("Top 200: move end date")
    axes[0, 0].set_ylabel("CBG Jaccard with main")
    axes[1, 0].set_ylabel("CBG Jaccard with main")
    axes[1, 0].set_xlabel("Date shift from main (days)")
    axes[1, 1].set_xlabel("Date shift from main (days)")
    for ax in axes.flat:
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(SHIFT_DAYS)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "figure_top100_top200_daily_window_overlap.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(overlap.sort_values(["min_away_obs", "window_name"]))
    print(
        top_destination_overlap.sort_values(
            ["top_n", "min_away_obs", "window_name"]
        )
    )


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine-factory',required=True,
                        help='Your authorized module:function returning the existing platform engine')
    parser.add_argument('--trajectory-table',required=True)
    parser.add_argument('--original-candidates',required=True)
    parser.add_argument('--result-prefix',required=True,
                        help='DATABASE.SCHEMA.PREFIX for four derived output tables')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    module,separator,function=args.engine_factory.partition(':')
    if not separator:
        parser.error('--engine-factory must be module:function')
    factory=getattr(importlib.import_module(module),function)
    run(factory(),args.trajectory_table,args.original_candidates,args.result_prefix,args.output)


if __name__=='__main__':
    main()
