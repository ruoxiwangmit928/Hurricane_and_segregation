"""Normalize saved weekly social-mixing levels without rebuilding their sample.

This is a downstream step: inputs must already contain the intended weekly
M (mixing) or S (segregation) estimates, after the study's cohort selection,
visit processing and aggregation. It does not average raw person/visit rows.
The change is 100*(M-M0)/(1-M0), equivalently -100*(S-S0)/S0.

Harvey example (one weekly aggregate per socioeconomic specification):
    python normalized_social_mixing_change.py --input weekly_social_mixing.csv \
      --value-column mean_social_mixing --scale mixing --time-column week \
      --time-type week --baseline-start 0 --baseline-end 5 \
      --group-columns label --output normalized_results

Ian example (already selected aggregate series, seven pre-event weeks):
    python normalized_social_mixing_change.py --input selected_ian_weekly.csv \
      --value-column mean_individual_seg --scale segregation \
      --time-column week_start_date --time-type date \
      --baseline-start 2022-08-05 --baseline-end 2022-09-16 \
      --group-columns cohort --output normalized_results

For community comparisons add --unit-column host_cbg: normalize each CBG
to its own baseline first, then report equal-CBG-weight mean/median changes.
Do not normalize a pooled CBG mean instead: different baselines generally
make those operations unequal. If inputs are already aggregate series,
omit --unit-column. Include every column defining a separate series among
--group-columns (for example cohort, threshold, or assignment draw).

Baseline weeks have equal weights. By default all specified baseline weeks
must be observed for every series. --minimum-baseline-weeks explicitly
permits a smaller observed baseline and records the support. Missing values
are never replaced by zero; a baseline M=1 has undefined normalized change
and raises an error. Source M and S levels are retained in the output.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd


def normalized_change(mixing, baseline_mixing):
    """Convert levels to complement-baseline changes; preserve missing levels."""
    mixing = np.asarray(mixing, dtype=float)
    baseline_mixing = np.asarray(baseline_mixing, dtype=float)
    if not np.isfinite(baseline_mixing).all() or np.any((baseline_mixing < 0) | (baseline_mixing >= 1)):
        raise ValueError("Baseline mixing must be finite and in [0,1); M0=1 is undefined")
    return 100 * (mixing - baseline_mixing) / (1 - baseline_mixing)


def normalize_weekly(frame, *, value_column, scale, time_column, time_type,
                     baseline_start, baseline_end, group_columns=(),
                     unit_column=None, minimum_baseline_weeks=None):
    groups = list(group_columns)
    series = groups + ([unit_column] if unit_column else [])
    if len(series) != len(set(series)) or time_column in series or value_column in series:
        raise ValueError("Series identifiers must be distinct from value/time columns")
    required = [*series, value_column, time_column]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Input is empty")
    out = frame.copy()
    if series and out[series].isna().any().any():
        raise ValueError("Missing series identifiers are not permitted")
    if time_type == "date":
        times = pd.to_datetime(out[time_column], errors="raise")
        if not times.eq(times.dt.normalize()).all():
            raise ValueError("Use dates marking the start of each week, without times")
        start, end = pd.Timestamp(baseline_start), pd.Timestamp(baseline_end)
        if end < start or (end-start).days % 7:
            raise ValueError("Baseline endpoints must delimit whole weekly steps")
        expected = list(pd.date_range(start, end, freq="7D"))
        if ((times-start).dt.days % 7).ne(0).any():
            raise ValueError("Input dates must share the baseline's weekly anchor")
    elif time_type == "week":
        times = pd.to_numeric(out[time_column], errors="raise")
        if times.isna().any() or not np.isfinite(times).all() or times.mod(1).ne(0).any():
            raise ValueError("Week indices must be finite integers")
        start, end = int(baseline_start), int(baseline_end)
        if str(start) != str(baseline_start) or str(end) != str(baseline_end) or end < start:
            raise ValueError("Baseline endpoints must be ordered integer weeks")
        expected = list(range(start, end+1))
        times = times.astype(int)
    else:
        raise ValueError("time_type must be week or date")
    if times.isna().any():
        raise ValueError("Missing week identifiers are not permitted")
    out[time_column] = times
    if out.duplicated([*series, time_column]).any():
        raise ValueError("Expected one preaggregated value per series and week; add missing grouping columns")
    values = pd.to_numeric(out[value_column], errors="raise")
    observed = values.dropna()
    if not np.isfinite(observed).all() or not observed.between(0, 1).all():
        raise ValueError("Observed M/S input values must be finite and within [0,1]")
    if scale not in {"mixing", "segregation"}:
        raise ValueError("scale must be mixing or segregation")
    out["mixing_level"] = values if scale == "mixing" else 1-values
    out["segregation_level"] = 1-out.mixing_level
    min_weeks = len(expected) if minimum_baseline_weeks is None else int(minimum_baseline_weeks)
    if not 1 <= min_weeks <= len(expected):
        raise ValueError("minimum_baseline_weeks must be between 1 and the baseline length")
    # A temporary grouping key supports a single already-aggregate series.
    key = "__normalization_single_series__"
    if key in out:
        raise ValueError(f"Reserved column: {key}")
    grouping = series or [key]
    if not series:
        out[key] = "all"
    baseline_rows = []
    normalized_parts = []
    for _, part in out.groupby(grouping, sort=False, dropna=False):
        identifiers = {c: part[c].iloc[0] for c in series}
        baseline = part.loc[part[time_column].isin(expected), "mixing_level"]
        n_observed = int(baseline.notna().sum())
        if n_observed < min_weeks:
            raise ValueError(f"Insufficient baseline for {identifiers or 'all'}: {n_observed}/{len(expected)} observed weeks")
        m0 = float(baseline.mean())
        part = part.copy()
        part["baseline_mixing"] = m0
        part["baseline_observed_weeks"] = n_observed
        part["normalized_change_percent"] = normalized_change(part.mixing_level, m0)
        # Explicit algebra check guards accidental division by baseline M.
        np.testing.assert_allclose(part.normalized_change_percent,
            -100*(part.segregation_level-(1-m0))/(1-m0), atol=1e-10, equal_nan=True)
        baseline_rows.append({**identifiers, "baseline_mixing": m0,
            "baseline_expected_weeks": len(expected), "baseline_observed_weeks": n_observed})
        normalized_parts.append(part)
    normalized = pd.concat(normalized_parts).sort_index()
    if not series:
        normalized = normalized.drop(columns=key)
    baselines = pd.DataFrame(baseline_rows)
    summaries = None
    if unit_column:
        keys = [*groups, time_column]
        summaries = normalized.groupby(keys, sort=True, dropna=False).agg(
            mean_normalized_change_percent=("normalized_change_percent", "mean"),
            median_normalized_change_percent=("normalized_change_percent", "median"),
            n_units_with_values=("normalized_change_percent", "count"),
            n_unit_rows=(unit_column, "size"),
        ).reset_index()
    return normalized, baselines, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--value-column", required=True)
    parser.add_argument("--scale", choices=["mixing", "segregation"], required=True)
    parser.add_argument("--time-column", required=True)
    parser.add_argument("--time-type", choices=["week", "date"], required=True)
    parser.add_argument("--baseline-start", required=True)
    parser.add_argument("--baseline-end", required=True)
    parser.add_argument("--group-columns", nargs="*", default=[])
    parser.add_argument("--unit-column")
    parser.add_argument("--minimum-baseline-weeks", type=int)
    args = parser.parse_args()
    identifiers = list(args.group_columns) + ([args.unit_column] if args.unit_column else [])
    frame = pd.read_csv(args.input, sep="\t" if args.input.suffix.lower() in {".tsv", ".txt"} else ",",
                        dtype={column: "string" for column in identifiers})
    normalized, baselines, summary = normalize_weekly(frame,
        value_column=args.value_column, scale=args.scale, time_column=args.time_column,
        time_type=args.time_type, baseline_start=args.baseline_start, baseline_end=args.baseline_end,
        group_columns=args.group_columns, unit_column=args.unit_column,
        minimum_baseline_weeks=args.minimum_baseline_weeks)
    args.output.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(args.output / "normalized_mixing_weekly.csv", index=False)
    baselines.to_csv(args.output / "baseline_summary.csv", index=False)
    if summary is not None:
        summary.to_csv(args.output / "group_normalized_change.csv", index=False)
    audit = dict(source=str(args.input), input_rows=len(frame), series=len(baselines),
        input_scale=args.scale, formula="100*(M-M0)/(1-M0)",
        baseline_start=args.baseline_start, baseline_end=args.baseline_end,
        baseline_weighting="equal observed weeks", group_columns=args.group_columns,
        unit_column=args.unit_column, missing_values_replaced=False,
        aggregation="per-unit baseline then equal-unit summaries" if args.unit_column else "already aggregated series; no reweighting",
        source_missing_values=int(normalized.mixing_level.isna().sum()))
    (args.output / "normalization_audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(f"Saved {len(normalized)} normalized rows and {len(baselines)} baseline summaries to {args.output}")


if __name__ == "__main__":
    main()
