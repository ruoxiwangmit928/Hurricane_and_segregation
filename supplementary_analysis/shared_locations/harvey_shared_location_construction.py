"""Construct Harvey fixed-reference shared-location sensitivity aggregates.

Inputs are existing observed visit rows and a prepared host-POI-week table:
    python harvey_shared_location_construction.py --evacuee-visits visits.csv \
        --host-activity host_poi_week_activity.csv --output shared_tables

Evacuee columns: device_id, place_id, date, start_date, end_date, and one of
stay_time/dwell_time/dwell_time_minutes. Host columns: week, place_id,
host_total_dwell, one row per active host POI per week. Full Aug25-Sep28
visit coverage and all 12 host weeks are required. This does not identify
evacuees, create host cohorts, or reconstruct raw GPS/POI matches.

Reference profiles are fixed across host weeks, count distinct evacuees,
include observed visits within each recorded displacement spell (inclusive
end dates), and use thresholds 1/3/5/10. Only aggregate tables are exported.
The original August-24-departure exclusion is retained as a diagnostic.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

STUDY_START = pd.Timestamp("2017-07-07")
STUDY_END = pd.Timestamp("2017-09-28")
ACUTE_START = pd.Timestamp("2017-08-25")
ACUTE_END = pd.Timestamp("2017-09-07")
THRESHOLDS = [1, 3, 5, 10]
BASELINE_WEEKS = list(range(6))
ACUTE = "Fixed weeks 7--8"
EXTENDED = "Fixed full displacement"  # Original label: observed Aug25–Sep28.
EXCLUDE_AUG24 = "Fixed weeks 7--8 excluding Aug 24 departures"
METRICS = ["host_dwell_shared", "host_dwell_nonshared", "encounter_index", "host_shared_ratio"]


def load_inputs(evacuee_path, host_path):
    columns = pd.read_csv(evacuee_path, nrows=0).columns
    dwell = next((c for c in ["stay_time", "dwell_time", "dwell_time_minutes"] if c in columns), None)
    if dwell is None:
        raise ValueError("Evacuee visits require stay_time, dwell_time or dwell_time_minutes")
    required = ["device_id", "place_id", "date", "start_date", "end_date", dwell]
    evac = pd.read_csv(evacuee_path, usecols=required,
                       dtype={"device_id": "string", "place_id": "string"}, low_memory=False)
    input_rows = len(evac)
    evac = evac.rename(columns={dwell: "evacuee_dwell"})
    for column in ["date", "start_date", "end_date"]:
        evac[column] = pd.to_datetime(evac[column], errors="coerce").dt.normalize()
    evac["evacuee_dwell"] = pd.to_numeric(evac.evacuee_dwell, errors="coerce")
    evac = evac.dropna(subset=["device_id", "place_id", "date", "start_date", "end_date", "evacuee_dwell"])
    valid_rows = len(evac)
    evac = evac.loc[
        evac.date.between(evac.start_date, evac.end_date, inclusive="both")
        & evac.date.between(ACUTE_START, STUDY_END, inclusive="both")
        & evac.evacuee_dwell.ge(0)
    ].copy()
    if evac.empty or evac.date.min() > ACUTE_START or evac.date.max() < STUDY_END:
        raise ValueError("Prepared evacuee visits do not cover Aug25–Sep28; obtain full observed visit rows")
    if not np.isfinite(evac.evacuee_dwell).all():
        raise ValueError("Nonfinite evacuee dwell")
    host = pd.read_csv(host_path, usecols=["week", "place_id", "host_total_dwell"], dtype={"place_id": "string"})
    host["host_total_dwell"] = pd.to_numeric(host.host_total_dwell, errors="coerce")
    host = host.dropna(subset=["week", "place_id", "host_total_dwell"])
    if host.duplicated(["week", "place_id"]).any():
        raise ValueError("Host input must have one row per active POI-week, not raw visit rows")
    if set(host.week) != set(range(12)) or not np.isfinite(host.host_total_dwell).all() or not host.host_total_dwell.ge(0).all():
        raise ValueError("Expected all twelve host weeks and finite nonnegative dwell")
    audit = dict(input_evacuee_rows=input_rows, rows_dropped_missing=input_rows - valid_rows,
                 retained_spell_and_reference_rows=len(evac), host_active_poi_week_rows=len(host))
    return evac, host, audit


def construct_tables(evac, host):
    acute = evac.loc[evac.date.between(ACUTE_START, ACUTE_END, inclusive="both")]
    definitions = {ACUTE: acute, EXCLUDE_AUG24: acute.loc[acute.start_date > pd.Timestamp("2017-08-24")], EXTENDED: evac}
    profiles = {name: rows.groupby("place_id", as_index=False).agg(
        evacuee_dwell=("evacuee_dwell", "sum"), unique_evacuees=("device_id", "nunique"))
        for name, rows in definitions.items()}
    rows = []
    for name, profile in profiles.items():
        for threshold in THRESHOLDS:
            exposure = profile.loc[profile.unique_evacuees.ge(threshold), ["place_id", "evacuee_dwell"]]
            total_evacuee_dwell = exposure.evacuee_dwell.sum()
            if total_evacuee_dwell <= 0:
                raise ValueError(f"No positive evacuee dwell for {name}, threshold {threshold}")
            for week, host_week in host.groupby("week"):
                joined = host_week.merge(exposure, on="place_id", how="left", validate="one_to_one")
                shared = joined.evacuee_dwell.notna()
                host_total = joined.host_total_dwell.sum()
                if host_total <= 0 or not shared.any() or shared.all():
                    raise ValueError(f"Missing positive/group support: {name}, threshold {threshold}, week {week}")
                rows.append(dict(definition=name, threshold=threshold, week=int(week),
                    host_dwell_shared=joined.loc[shared, "host_total_dwell"].mean(),
                    host_dwell_nonshared=joined.loc[~shared, "host_total_dwell"].mean(),
                    encounter_index=(joined.loc[shared, "host_total_dwell"] * joined.loc[shared, "evacuee_dwell"]).sum() / (host_total * total_evacuee_dwell),
                    host_shared_ratio=joined.loc[shared, "host_total_dwell"].sum() / host_total))
    weekly = pd.DataFrame(rows)
    keys = pd.MultiIndex.from_frame(weekly[["definition", "threshold"]])
    for metric in METRICS:
        baseline = weekly.loc[weekly.week.isin(BASELINE_WEEKS)].groupby(["definition", "threshold"])[metric].mean()
        values = baseline.reindex(keys).to_numpy()
        if not (values > 0).all():
            raise ValueError(f"Nonpositive baseline for {metric}")
        weekly[f"change_{metric}_pct"] = 100 * (weekly[metric].to_numpy() - values) / values
    phase = weekly.assign(period=np.select(
        [weekly.week.isin([7, 8]), weekly.week.isin([9, 10, 11])],
        ["weeks 7--8", "weeks 9--11"], default="other"))
    phase = phase.loc[phase.period.ne("other")].groupby(["definition", "threshold", "period"], as_index=False)[[f"change_{m}_pct" for m in METRICS]].mean()
    comparisons = []
    for threshold in THRESHOLDS:
        for a, b, label in [(ACUTE, EXTENDED, "weeks 7--8 vs full"), (EXCLUDE_AUG24, ACUTE, "exclude vs include Aug 24 departures")]:
            set_a = set(profiles[a].loc[profiles[a].unique_evacuees.ge(threshold), "place_id"])
            set_b = set(profiles[b].loc[profiles[b].unique_evacuees.ge(threshold), "place_id"])
            comparisons.append(dict(comparison=label, threshold=threshold, locations_a=len(set_a), locations_b=len(set_b),
                intersection=len(set_a & set_b), only_a=len(set_a - set_b), only_b=len(set_b - set_a),
                jaccard=len(set_a & set_b) / len(set_a | set_b)))
    locations = pd.DataFrame(comparisons)
    overlap = locations.loc[locations.comparison.eq("weeks 7--8 vs full")].drop(columns="comparison").rename(columns={
        "locations_a": "acute_shared_pois", "locations_b": "full_displacement_shared_pois",
        "only_a": "acute_only", "only_b": "full_only"})
    overlap["acute_retained_in_full"] = overlap.intersection / overlap.acute_shared_pois
    fixed = weekly.loc[weekly.definition.isin([ACUTE, EXTENDED])].copy()
    fixed["week_start"] = STUDY_START + pd.to_timedelta(7 * fixed.week, unit="D")
    fixed = fixed.rename(columns={"host_dwell_shared": "mean_dwell_shared", "host_dwell_nonshared": "mean_dwell_nonshared",
        "change_host_dwell_shared_pct": "pct_change_mean_dwell_shared", "change_host_dwell_nonshared_pct": "pct_change_mean_dwell_nonshared",
        "change_encounter_index_pct": "pct_change_encounter_index", "change_host_shared_ratio_pct": "pct_change_host_shared_ratio"})
    coverage = pd.DataFrame([dict(evacuee_source="supplied observed visit file", observed_start=evac.date.min().date(),
        observed_end=evac.date.max().date(), devices=evac.device_id.nunique(), acute_visit_rows=len(acute),
        full_displacement_visit_rows=len(evac))])
    return {
        "comment7_fixed_definition_weekly_metrics.csv": weekly,
        "comment7_fixed_definition_phase_summary.csv": phase,
        "comment7_fixed_definition_location_comparison.csv": locations,
        "comment7_fixed_period_definition_weekly.csv": fixed,
        "comment7_fixed_period_definition_overlap.csv": overlap,
        "comment7_fixed_period_definition_coverage.csv": coverage,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evacuee-visits", required=True, type=Path)
    parser.add_argument("--host-activity", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    evac, host, audit = load_inputs(args.evacuee_visits, args.host_activity)
    tables = construct_tables(evac, host)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(args.output / name, index=False)
    audit.update(input_stage="Existing complete observed evacuee visits and host-POI-week dwell",
                 acute_reference=[str(ACUTE_START.date()), str(ACUTE_END.date())],
                 extended_reference=[str(ACUTE_START.date()), str(STUDY_END.date())],
                 thresholds=THRESHOLDS, baseline_weeks=BASELINE_WEEKS,
                 spell_end_rule="inclusive, matching the original shared-location analysis",
                 main_threshold=3, exported_device_or_poi_records=False)
    (args.output / "construction_summary.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Saved six aggregate tables to {args.output}")


if __name__ == "__main__":
    main()
