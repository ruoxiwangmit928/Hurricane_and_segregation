"""Current Ian mixing from the prepared, Florida-bounded analysis tables.

This is the threshold-sensitivity calculation underlying the current aggregate
Ian results: August 5--October 27, 2022, in Friday-starting weeks; baseline is
the seven weeks before September 23. Fixed target locations require 1/3/5/10
distinct evacuees during September 23--October 10, within identified episodes.
The main comparison selects threshold 3, Top-100, all_by_received strict stayers.

Unlike the two legacy generic M scripts, this program reuses the established
host/evacuee labels, income assignments and place-week visitor composition. It
does not apply the legacy monthly-ten-day active-user screen, rebuild homes,
reidentify evacuees, rescan raw GPS, or redefine the measurement universe.
Do not substitute the separately rescanned GE3EVAC tables for the BOUNDED
inputs: the GE3EVAC tables belong to the separate income-imputation analysis.

Required prepared tables and columns:
  host_visits: host_definition, host_rule, host_rank, cuebiq_id, place_id,
      event_date (YYYYMMDD integer), admin1_id, dwell_time_minutes.
  evacuee_visits: same except host_rule; also is_during_displacement.
  analysis_devices: cuebiq_id (union of original host/evacuee populations).
  bounded_visits: cuebiq_id, place_id, event_date, visit_date, week_start_date,
      dwell_time_minutes; original deduplicated FL-bounded all-visitor visits
      for Aug 5--Oct 27, with Fridays as week starts and dwell 10--240 minutes.
  visitor_income: cuebiq_id from the original BOUNDED income assignments.
  place_week: place_id, week_start_date, q1_share..q4_share,
      place_income_segregation from the original BOUNDED place-week analysis.

Supply an authorized engine with execute_statement(sql) and read_sql(sql),
or --engine-factory module:function returning that engine. Six derived SQL
tables are created/replaced under --result-prefix; inputs are never written.
--sql-only writes the SQL for inspection without connecting or executing it.
The local CSV outputs are aggregate only; individual intermediate records
stay in the authorized database. Normalized mixing is 100*(M-M0)/(1-M0).
"""
from pathlib import Path
import argparse
import importlib
import json
import re

import numpy as np
import pandas as pd

INPUT_NAMES = ("host_visits", "evacuee_visits", "analysis_devices", "bounded_visits", "visitor_income", "place_week")
OUTPUT_SUFFIXES = {
    "target_pois": "TARGET_POIS", "case_visits": "CASE_VISITS",
    "comparison_visits": "COMPARISON_VISITS", "all_visits": "ALL_VISITS",
    "individual_week": "INDIVIDUAL_WEEK", "weekly_results": "WEEKLY_RESULTS",
}
KEYS = ["min_evac_devices", "cohort", "host_definition", "host_rule", "top_n"]
WEEKS = pd.date_range("2022-08-05", periods=12, freq="7D")
EVENT_START = pd.Timestamp("2022-09-23")


def identifier(value):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError("Use an unquoted DATABASE.SCHEMA.TABLE identifier")
    return value.upper()


def table_names(inputs, result_prefix):
    if set(inputs) != set(INPUT_NAMES):
        raise ValueError(f"Expected exactly these input tables: {INPUT_NAMES}")
    tables = {key: identifier(value) for key, value in inputs.items()}
    prefix = identifier(result_prefix)
    outputs = {key: identifier(prefix + "_" + suffix) for key, suffix in OUTPUT_SUFFIXES.items()}
    if set(outputs.values()) & set(tables.values()):
        raise ValueError("Output table collides with an input table")
    tables.update(outputs)
    return tables


# Extracted SQL retains the original thresholds, cohorts and aggregation.
# Only table identifiers are substituted below; these are validated above.
SQL_TEMPLATES = [
    ('execute_statement', """
CREATE OR REPLACE TABLE {target_pois} AS
WITH thresholds AS (
    SELECT COLUMN1::NUMBER AS min_evac_devices
    FROM VALUES (1), (3), (5), (10)
),

top_sizes AS (
    SELECT COLUMN1::NUMBER AS top_n
    FROM VALUES (50), (100), (200)
),

evac_event_pois AS (
    SELECT
        E.host_definition,
        T.top_n,
        E.place_id,
        COUNT(DISTINCT E.cuebiq_id) AS n_evac_devices_event,
        COUNT(*) AS n_evac_visits_event,
        SUM(E.dwell_time_minutes) AS evac_dwell_event
    FROM {evacuee_visits} E
    JOIN top_sizes T
      ON E.host_rank <= T.top_n
    WHERE E.event_date BETWEEN 20220923 AND 20221010
      AND E.place_id IS NOT NULL
      AND E.admin1_id = 'US.FL'
      AND E.is_during_displacement = 1
      AND E.dwell_time_minutes BETWEEN 10 AND 240
    GROUP BY E.host_definition, T.top_n, E.place_id
),

eligible_event_pois AS (
    SELECT
        X.min_evac_devices,
        P.*
    FROM evac_event_pois P
    JOIN thresholds X
      ON P.n_evac_devices_event >= X.min_evac_devices
),

host_pois AS (
    SELECT
        P.min_evac_devices,
        H.host_definition,
        H.host_rule,
        T.top_n,
        H.place_id,
        1 AS visited_by_host,
        0 AS visited_by_evac,
        P.n_evac_devices_event,
        P.n_evac_visits_event,
        P.evac_dwell_event
    FROM {host_visits} H
    JOIN top_sizes T
      ON H.host_rank <= T.top_n
    JOIN eligible_event_pois P
      ON H.host_definition = P.host_definition
     AND T.top_n = P.top_n
     AND H.place_id = P.place_id
    WHERE H.event_date BETWEEN 20220805 AND 20221027
      AND H.place_id IS NOT NULL
      AND H.admin1_id = 'US.FL'
      AND H.dwell_time_minutes BETWEEN 10 AND 240
    GROUP BY
        P.min_evac_devices,
        H.host_definition,
        H.host_rule,
        T.top_n,
        H.place_id,
        P.n_evac_devices_event,
        P.n_evac_visits_event,
        P.evac_dwell_event
),

evac_pois AS (
    SELECT
        min_evac_devices,
        host_definition,
        'evacuee' AS host_rule,
        top_n,
        place_id,
        0 AS visited_by_host,
        1 AS visited_by_evac,
        n_evac_devices_event,
        n_evac_visits_event,
        evac_dwell_event
    FROM eligible_event_pois
)

SELECT
    min_evac_devices,
    host_definition,
    host_rule,
    top_n,
    place_id,
    MAX(visited_by_host) AS visited_by_host,
    MAX(visited_by_evac) AS visited_by_evac,
    MAX(n_evac_devices_event) AS n_evac_devices_event,
    MAX(n_evac_visits_event) AS n_evac_visits_event,
    MAX(evac_dwell_event) AS evac_dwell_event
FROM (
    SELECT * FROM host_pois
    UNION ALL
    SELECT * FROM evac_pois
)
GROUP BY
    min_evac_devices,
    host_definition,
    host_rule,
    top_n,
    place_id
"""),
    ('read_sql', """
SELECT
    min_evac_devices,
    host_definition,
    host_rule,
    top_n,
    COUNT(DISTINCT place_id) AS n_target_pois,
    MIN(n_evac_devices_event) AS min_evac_devices_event,
    MEDIAN(n_evac_devices_event) AS median_evac_devices_event,
    MAX(n_evac_devices_event) AS max_evac_devices_event
FROM {target_pois}
GROUP BY min_evac_devices, host_definition, host_rule, top_n
ORDER BY min_evac_devices, host_definition, host_rule, top_n
"""),
    ('execute_statement', """
CREATE OR REPLACE TABLE {case_visits} AS
WITH target_pairs AS (
    SELECT DISTINCT
        min_evac_devices,
        host_definition,
        top_n,
        place_id
    FROM {target_pois}
),

top_sizes AS (
    SELECT COLUMN1::NUMBER AS top_n
    FROM VALUES (50), (100), (200)
),

host_visits AS (
    SELECT
        P.min_evac_devices,
        'host' AS cohort,
        H.host_definition,
        H.host_rule,
        T.top_n,
        H.cuebiq_id,
        H.place_id,
        H.event_date,
        TO_DATE(H.event_date::VARCHAR, 'YYYYMMDD') AS visit_date,
        DATEADD(
            DAY,
            -MOD(DAYOFWEEKISO(TO_DATE(H.event_date::VARCHAR, 'YYYYMMDD')) - 5 + 7, 7),
            TO_DATE(H.event_date::VARCHAR, 'YYYYMMDD')
        ) AS week_start_date,
        H.dwell_time_minutes
    FROM {host_visits} H
    JOIN top_sizes T
      ON H.host_rank <= T.top_n
    JOIN target_pairs P
      ON H.host_definition = P.host_definition
     AND T.top_n = P.top_n
     AND H.place_id = P.place_id
    WHERE H.event_date BETWEEN 20220805 AND 20221027
      AND H.admin1_id = 'US.FL'
      AND H.dwell_time_minutes BETWEEN 10 AND 240
),

evac_visits AS (
    SELECT
        P.min_evac_devices,
        'evacuee' AS cohort,
        E.host_definition,
        'evacuee' AS host_rule,
        T.top_n,
        E.cuebiq_id,
        E.place_id,
        E.event_date,
        TO_DATE(E.event_date::VARCHAR, 'YYYYMMDD') AS visit_date,
        DATEADD(
            DAY,
            -MOD(DAYOFWEEKISO(TO_DATE(E.event_date::VARCHAR, 'YYYYMMDD')) - 5 + 7, 7),
            TO_DATE(E.event_date::VARCHAR, 'YYYYMMDD')
        ) AS week_start_date,
        E.dwell_time_minutes
    FROM {evacuee_visits} E
    JOIN top_sizes T
      ON E.host_rank <= T.top_n
    JOIN target_pairs P
      ON E.host_definition = P.host_definition
     AND T.top_n = P.top_n
     AND E.place_id = P.place_id
    WHERE E.event_date BETWEEN 20220805 AND 20221027
      AND E.admin1_id = 'US.FL'
      AND E.dwell_time_minutes BETWEEN 10 AND 240
)

SELECT * FROM host_visits
UNION ALL
SELECT * FROM evac_visits
"""),
    ('execute_statement', """
CREATE OR REPLACE TABLE {comparison_visits} AS
WITH target_pairs AS (
    SELECT DISTINCT
        min_evac_devices,
        host_definition,
        top_n,
        place_id
    FROM {target_pois}
),

case_ids AS (
    SELECT DISTINCT cuebiq_id
    FROM {analysis_devices}
)

SELECT
    T.min_evac_devices,
    'comparison' AS cohort,
    T.host_definition,
    'non_host_non_evacuee' AS host_rule,
    T.top_n,
    V.cuebiq_id,
    V.place_id,
    V.event_date,
    V.visit_date,
    V.week_start_date,
    V.dwell_time_minutes
FROM {bounded_visits} V
JOIN target_pairs T
  ON V.place_id = T.place_id
JOIN {visitor_income} Q
  ON V.cuebiq_id = Q.cuebiq_id
LEFT JOIN case_ids C
  ON V.cuebiq_id = C.cuebiq_id
WHERE C.cuebiq_id IS NULL
"""),
    ('execute_statement', """
CREATE OR REPLACE TABLE {all_visits} AS
SELECT *
FROM {case_visits}
UNION ALL
SELECT *
FROM {comparison_visits}
"""),
    ('execute_statement', """
CREATE OR REPLACE TABLE {individual_week} AS
WITH valid_visits AS (
    SELECT
        C.min_evac_devices,
        C.cohort,
        C.host_definition,
        C.host_rule,
        C.top_n,
        C.cuebiq_id,
        C.week_start_date,
        C.place_id,
        C.dwell_time_minutes,
        P.q1_share,
        P.q2_share,
        P.q3_share,
        P.q4_share,
        P.place_income_segregation
    FROM {all_visits} C
    JOIN {place_week} P
      ON C.place_id = P.place_id
     AND C.week_start_date = P.week_start_date
),

person_week AS (
    SELECT
        min_evac_devices,
        cohort,
        host_definition,
        host_rule,
        top_n,
        cuebiq_id,
        week_start_date,
        COUNT(*) AS n_valid_visits,
        COUNT(DISTINCT place_id) AS n_valid_pois,
        SUM(dwell_time_minutes) AS valid_dwell_minutes,
        SUM(dwell_time_minutes * q1_share) / NULLIF(SUM(dwell_time_minutes), 0) AS exp_q1,
        SUM(dwell_time_minutes * q2_share) / NULLIF(SUM(dwell_time_minutes), 0) AS exp_q2,
        SUM(dwell_time_minutes * q3_share) / NULLIF(SUM(dwell_time_minutes), 0) AS exp_q3,
        SUM(dwell_time_minutes * q4_share) / NULLIF(SUM(dwell_time_minutes), 0) AS exp_q4,
        SUM(dwell_time_minutes * place_income_segregation)
            / NULLIF(SUM(dwell_time_minutes), 0) AS avg_place_seg_exposure
    FROM valid_visits
    GROUP BY
        min_evac_devices,
        cohort,
        host_definition,
        host_rule,
        top_n,
        cuebiq_id,
        week_start_date
)

SELECT
    *,
    (2.0 / 3.0) * (
        ABS(COALESCE(exp_q1, 0) - 0.25) +
        ABS(COALESCE(exp_q2, 0) - 0.25) +
        ABS(COALESCE(exp_q3, 0) - 0.25) +
        ABS(COALESCE(exp_q4, 0) - 0.25)
    ) AS individual_income_segregation
FROM person_week
WHERE n_valid_visits >= 1
  AND n_valid_pois >= 1
"""),
    ('execute_statement', """
CREATE OR REPLACE TABLE {weekly_results} AS
WITH weekly AS (
    SELECT
        min_evac_devices,
        cohort,
        host_definition,
        host_rule,
        top_n,
        week_start_date,
        COUNT(DISTINCT cuebiq_id) AS n_devices,
        AVG(individual_income_segregation) AS mean_individual_seg,
        MEDIAN(individual_income_segregation) AS median_individual_seg,
        AVG(avg_place_seg_exposure) AS mean_place_seg_exposure,
        MEDIAN(avg_place_seg_exposure) AS median_place_seg_exposure,
        AVG(n_valid_visits) AS mean_valid_visits,
        AVG(n_valid_pois) AS mean_valid_pois
    FROM {individual_week}
    GROUP BY
        min_evac_devices,
        cohort,
        host_definition,
        host_rule,
        top_n,
        week_start_date
),

baseline AS (
    SELECT
        min_evac_devices,
        cohort,
        host_definition,
        host_rule,
        top_n,
        AVG(mean_individual_seg) AS baseline_mean_individual_seg,
        AVG(mean_place_seg_exposure) AS baseline_mean_place_seg_exposure
    FROM weekly
    WHERE week_start_date < TO_DATE('2022-09-23')
    GROUP BY
        min_evac_devices,
        cohort,
        host_definition,
        host_rule,
        top_n
)

SELECT
    W.*,
    B.baseline_mean_individual_seg,
    B.baseline_mean_place_seg_exposure,

    W.mean_individual_seg - B.baseline_mean_individual_seg
        AS delta_mean_individual_seg,

    100 * (W.mean_individual_seg - B.baseline_mean_individual_seg)
        / NULLIF(B.baseline_mean_individual_seg, 0)
        AS pct_change_mean_individual_seg,

    W.mean_place_seg_exposure - B.baseline_mean_place_seg_exposure
        AS delta_mean_place_seg_exposure,

    100 * (W.mean_place_seg_exposure - B.baseline_mean_place_seg_exposure)
        / NULLIF(B.baseline_mean_place_seg_exposure, 0)
        AS pct_change_mean_place_seg_exposure
FROM weekly W
JOIN baseline B
  ON W.min_evac_devices = B.min_evac_devices
 AND W.cohort = B.cohort
 AND W.host_definition = B.host_definition
 AND W.host_rule = B.host_rule
 AND W.top_n = B.top_n
"""),
    ('read_sql', """
SELECT *
FROM {weekly_results}
ORDER BY min_evac_devices, cohort, host_definition, host_rule, top_n, week_start_date
"""),
]


def build_queries(inputs, result_prefix):
    tables = table_names(inputs, result_prefix)
    return [(mode, template.format_map(tables)) for mode, template in SQL_TEMPLATES]


def normalize_weekly(frame):
    """Keep original S fields and append absolute M plus normalized changes."""
    frame = frame.copy()
    frame.columns = frame.columns.str.lower()
    required = set(KEYS + ["week_start_date", "mean_individual_seg", "baseline_mean_individual_seg", "pct_change_mean_individual_seg"])
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing result columns: {sorted(required - set(frame.columns))}")
    frame["week_start_date"] = pd.to_datetime(frame.week_start_date)
    if frame.duplicated(KEYS + ["week_start_date"]).any():
        raise ValueError("Duplicate cohort/specification-week rows")
    if not frame.mean_individual_seg.between(0, 1).all():
        raise ValueError("Expected finite individual segregation means in [0, 1]")
    if not frame.week_start_date.isin(WEEKS).all():
        raise ValueError("Expected Friday weeks from Aug 5 through Oct 21")
    rows = []
    for key, g in frame.groupby(KEYS, sort=False, dropna=False):
        g = g.sort_values("week_start_date").copy()
        pre = g.week_start_date.lt(EVENT_START)
        if set(g.loc[pre, "week_start_date"]) != set(WEEKS[:7]):
            raise ValueError(f"Incomplete seven-week baseline for {key}")
        baseline_s = g.loc[pre, "mean_individual_seg"].mean()
        if not np.isfinite(baseline_s) or baseline_s <= 0:
            raise ValueError(f"Normalized mixing undefined at zero segregation baseline: {key}")
        np.testing.assert_allclose(g.baseline_mean_individual_seg, baseline_s, atol=1e-9, rtol=0)
        g["mean_social_mixing"] = 1 - g.mean_individual_seg
        g["baseline_social_mixing"] = 1 - baseline_s
        g["normalized_change_pct"] = -100 * (g.mean_individual_seg - baseline_s) / baseline_s
        np.testing.assert_allclose(g.normalized_change_pct, -g.pct_change_mean_individual_seg, atol=1e-8, rtol=0)
        rows.append(g)
    if not rows:
        raise ValueError("No weekly result rows")
    return pd.concat(rows, ignore_index=True)


def run(engine, inputs, result_prefix, output):
    tables = table_names(inputs, result_prefix)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    read_results = []
    for mode, query in build_queries(inputs, result_prefix):
        if mode == "execute_statement":
            engine.execute_statement(query)
        else:
            result = engine.read_sql(query)
            result.columns = result.columns.str.lower()
            read_results.append(result)
    location_counts, original_weekly = read_results
    weekly = normalize_weekly(original_weekly)
    location_counts.to_csv(output / "ian_target_location_counts.csv", index=False)
    weekly.to_csv(output / "ian_mixing_weekly.csv", index=False)
    main = weekly.loc[weekly.min_evac_devices.eq(3) & weekly.top_n.eq(100)
                      & weekly.host_definition.eq("all_by_received")
                      & ((weekly.cohort.eq("host") & weekly.host_rule.eq("strict_stayer"))
                         | weekly.cohort.isin(["evacuee", "comparison"]))].copy()
    if len(main) != 36 or main.cohort.nunique() != 3:
        raise ValueError("Expected 12 weeks in each of the three main Top-100 cohorts")
    main.to_csv(output / "ian_mixing_main_top100.csv", index=False)
    audit = {
        "analysis_dates_inclusive": ["2022-08-05", "2022-10-27"],
        "fixed_reference_dates_inclusive": ["2022-09-23", "2022-10-10"],
        "baseline_week_starts": [str(x.date()) for x in WEEKS[:7]],
        "thresholds": [1, 3, 5, 10], "main_threshold": 3,
        "top_n_values": [50, 100, 200],
        "normalization": "100*(M-M0)/(1-M0) = -100*(S-S0)/S0",
        "inputs": {key: tables[key] for key in INPUT_NAMES},
        "derived_tables": {key: tables[key] for key in OUTPUT_SUFFIXES},
        "upstream_measurement_universe": "Existing Florida BOUNDED tables, not GE3EVAC-rescanned tables",
        "upstream_cohorts_preserved": True,
        "normalized_weekly_rows": len(weekly), "main_rows": len(main),
        "device_records_exported_locally": False,
    }
    (output / "ian_mixing_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return weekly


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for key in INPUT_NAMES:
        parser.add_argument("--" + key.replace("_", "-"), required=True,
                            help="Original prepared table: DATABASE.SCHEMA.TABLE")
    parser.add_argument("--result-prefix", required=True, help="DATABASE.SCHEMA.NEW_PREFIX for six derived tables")
    parser.add_argument("--output", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--engine-factory", help="Importable module:function returning an authorized engine")
    mode.add_argument("--sql-only", action="store_true", help="Write SQL without executing database operations")
    args = parser.parse_args()
    inputs = {key: getattr(args, key) for key in INPUT_NAMES}
    queries = build_queries(inputs, args.result_prefix)
    if args.sql_only:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "ian_mixing_queries.sql").write_text(
            "\n\n".join(query.strip() + ";" for _, query in queries) + "\n")
        print(f"Wrote SQL only to {args.output}; no database operations executed")
        return
    module_name, separator, function_name = args.engine_factory.partition(":")
    if not separator or not module_name or not function_name:
        parser.error("--engine-factory must be module:function")
    engine = getattr(importlib.import_module(module_name), function_name)()
    run(engine, inputs, args.result_prefix, args.output)
    print(f"Saved aggregate Ian results to {args.output}")


if __name__ == "__main__":
    main()
