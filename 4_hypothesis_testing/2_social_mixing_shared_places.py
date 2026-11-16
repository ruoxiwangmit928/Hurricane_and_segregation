"""Harvey shared-location mixing, travel distance and categories (Figure 4d/e).

Both groups use the fixed Aug25--Sep7 >=3-distinct-evacuee reference, including
only visits within recorded displacement episodes. Non-shared includes POIs
with one/two evacuees. The place-mixing snapshot stays on absolute M=1-S.

Two ways to supply the week-9 boxplots:
  --boxplot-export DIR  reads verified aggregate audit.json and
                       fig4d_boxplot_statistics.json from the server; or
  --place-change FILE --daily-visits DIR --hosts FILE [--distance-unit m]
                       computes those statistics from the original inputs.

All runs also require --evacuee-visits FILE --host-activity FILE
--poi-metadata FILE --taxonomy FILE --output DIR. Use --no-figures for tables.
The host activity table contains week/place_id/host_total_dwell; metadata uses
safegraph_place_id/top_category/sub_category, taxonomy uses those categories
and Taxonomy. Category shares use week 8 (Sep1--7), one classified POI each.

Raw boxplot mode requires headerless place_change_rate.txt columns listed in
PLACE_COLUMNS, the original one-ID-per-line host roster, and all seven daily
CSV files 2017-09-08.csv through 2017-09-14.csv. Their columns are device_id,
place_id, travel_distance. No missing day is silently skipped. Place mixing
uses n>10 and a pooled 1.5-IQR screen on S. Travel keeps each valid host visit,
converts metres to kilometres, and hides rather than removes boxplot fliers.
No individual/device-level outputs are exported.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from shared_location_reference import (build_reference, read_host_activity,
    clean_ids, plotting, save_plot, GROUPS, COLORS)

PLACE_COLUMNS = ["place_id", "n", "q1", "q2", "q3", "q4",
                 "segregation_value", "week", "change_rate"]
TARGET_START = "2017-09-08"
TARGET_END = "2017-09-14"
TARGET_WEEK = 9
CATEGORY_WEEK = 8
MIN_PLACE_N = 10  # n > 10, not >=10
DISPLAY_CATEGORIES = ["City / Outdoors", "Coffee / Tea", "College", "Entertainment",
    "Food", "Gasoline", "Grocery", "Health", "Hotel", "Service", "Shopping", "Sports"]

def require_columns(path, columns):
    available = pd.read_csv(path, nrows=0).columns.tolist()
    missing = sorted(set(columns) - set(available))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; found {available}")

def mixing_values(path, shared, target_week, min_place_n):
    # Headerless original file; its first field stores a POI ID despite being
    # named device_id in the old notebook. It is NOT an individual ID here.
    raw = pd.read_csv(path, names=PLACE_COLUMNS, dtype={"place_id": "string"},
                      low_memory=False)
    for col in ["n", "week", "segregation_value"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw["place_id"] = clean_ids(raw["place_id"])
    week = raw.loc[raw.week.eq(target_week) & raw.n.gt(min_place_n)].copy()
    n_missing = int(week[["place_id", "segregation_value"]].isna().any(axis=1).sum())
    week = week.dropna(subset=["place_id", "segregation_value"])
    if week.empty:
        raise ValueError("No eligible week-9 place values: check the headerless place file and n > 10.")
    if week.place_id.duplicated().any():
        raise ValueError("Duplicate place IDs in the selected week: expected one row per POI-week.")
    if not np.isfinite(week.segregation_value).all() or not week.segregation_value.between(0, 1).all():
        raise ValueError("Selected segregation values must be finite and in [0,1].")
    # Original final cell: ONE pooled IQR screen across both groups, on S.
    q1, q3 = week.segregation_value.quantile([0.25, 0.75]).to_numpy()
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    keep = week.segregation_value.between(lo, hi)
    kept = week.loc[keep].copy()
    kept["group"] = np.where(kept.place_id.isin(shared), "Shared", "Non-shared")
    values = {group: (1.0 - kept.loc[kept.group.eq(group), "segregation_value"]).to_numpy()
              for group in GROUPS}
    before_groups = np.where(week.place_id.isin(shared), "Shared", "Non-shared")
    before_iqr = {}
    for group in GROUPS:
        s = week.loc[before_groups == group, "segregation_value"]
        before_iqr[group] = {"n_pois": len(s),
                             "mean_mixing": float((1 - s).mean()) if len(s) else None}
    audit = {"target_week": int(target_week), "minimum_n_rule": f"n > {min_place_n}",
             "eligible_before_missing_drop": int(len(week) + n_missing),
             "dropped_missing_place_or_S": n_missing,
             "eligible_before_pooled_IQR": len(week),
             "pooled_S_Q1": float(q1), "pooled_S_Q3": float(q3),
             "pooled_S_lower_fence": float(lo), "pooled_S_upper_fence": float(hi),
             "removed_by_pooled_IQR": int((~keep).sum()),
             "retained_pois": len(kept),
             "before_IQR_group_means_for_C7_comparison_only": before_iqr,
             "statistical_unit": "one equally weighted POI in week 9",
             "outcome": "M = 1 - S (absolute place-level mixing, not a change rate)"}
    return values, audit

def travel_values(daily_files, host_file, shared, chunksize, unit):
    if unit not in {"m", "km"}:
        raise ValueError("TRAVEL_DISTANCE_INPUT_UNIT must be 'm' or 'km'.")
    roster = pd.read_csv(host_file, header=None, dtype="string")
    if roster.shape[1] != 1:
        raise ValueError("HOST_FILE must contain one original host ID per line and no header.")
    ids = clean_ids(roster.iloc[:, 0]).dropna()
    hosts = set(ids)
    if not hosts:
        raise ValueError("The host roster is empty.")
    parts = {group: [] for group in GROUPS}
    poi_sets = {group: set() for group in GROUPS}
    daily_audit = []
    factor = 1000.0 if unit == "m" else 1.0
    for path in daily_files:
        require_columns(path, ["device_id", "place_id", "travel_distance"])
        counts = {"date": path.stem, "source_rows": 0, "host_rows": 0,
                  "dropped_missing_place": 0, "dropped_invalid_distance": 0,
                  "retained_shared_visits": 0, "retained_nonshared_visits": 0}
        for chunk in pd.read_csv(path, usecols=["device_id", "place_id", "travel_distance"],
                                 dtype={"device_id": "string", "place_id": "string"},
                                 chunksize=chunksize, low_memory=False):
            counts["source_rows"] += len(chunk)
            host = chunk.loc[clean_ids(chunk.device_id).isin(hosts)].copy()
            counts["host_rows"] += len(host)
            host["place_id"] = clean_ids(host.place_id)
            counts["dropped_missing_place"] += int(host.place_id.isna().sum())
            host = host.dropna(subset=["place_id"])
            distance = pd.to_numeric(host.travel_distance, errors="coerce")
            finite = np.isfinite(distance)
            counts["dropped_invalid_distance"] += int((~finite).sum())
            if (distance.loc[finite] < 0).any():
                raise ValueError(f"Negative travel_distance in {path}; inspect the source, do not silently trim.")
            host = host.loc[finite]
            distance = distance.loc[finite] / factor
            shared_mask = host.place_id.isin(shared)
            for group, mask in [("Shared", shared_mask), ("Non-shared", ~shared_mask)]:
                parts[group].append(distance.loc[mask].to_numpy(dtype=float))
                poi_sets[group].update(host.loc[mask, "place_id"])
                key = "retained_shared_visits" if group == "Shared" else "retained_nonshared_visits"
                counts[key] += int(mask.sum())
        daily_audit.append(counts)
        print(f"Read {path.name}: {counts['host_rows']:,} host visits")
    values = {group: np.concatenate(parts[group]) if parts[group] else np.array([])
              for group in GROUPS}
    audit = {"original_host_roster_distinct_ids": len(hosts),
             "original_host_roster_duplicate_lines": int(len(ids) - len(hosts)),
             "input_distance_unit": unit, "export_distance_unit": "km",
             "statistical_unit": "one observed host visit; no POI or person averaging",
             "distance_outlier_removal": "none; outlier markers hidden only",
             "daily_counts": daily_audit,
             "n_active_pois": {group: len(poi_sets[group]) for group in GROUPS}}
    return values, audit

def boxplot_summary(values, metric, group, unit):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError(f"{metric}, {group}: expected a nonempty finite one-dimensional sample.")
    # Matches default NumPy/Matplotlib/Seaborn quartiles and 1.5-IQR whiskers.
    q1, median, q3 = np.percentile(x, [25, 50, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    in_low = x[x >= lower]
    in_high = x[x <= upper]
    whislo = min(float(q1), float(in_low.min())) if len(in_low) else float(q1)
    whishi = max(float(q3), float(in_high.max())) if len(in_high) else float(q3)
    return {"metric": metric, "group": group, "unit": unit, "n": int(len(x)),
            "mean": float(x.mean()), "q1": float(q1), "median": float(median),
            "q3": float(q3), "whislo": whislo, "whishi": whishi,
            "n_below_whisker": int((x < whislo).sum()),
            "n_above_whisker": int((x > whishi).sum()),
            "whisker_rule": "1.5 IQR", "showfliers": False}

def validate_boxplots(stats, audit, reference):
    expected_days = [d.strftime("%Y-%m-%d") for d in pd.date_range(TARGET_START, TARGET_END)]
    required = {
        "shared_threshold_distinct_people": 3, "reference_start": "2017-08-25",
        "reference_end_inclusive": "2017-09-07", "target_start": TARGET_START,
        "target_end_inclusive": TARGET_END}
    for key, expected in required.items():
        if audit.get(key) != expected:
            raise ValueError(f"Boxplot export {key}: expected {expected!r}")
    if audit["shared_reference"]["shared_set_sha256"] != reference["shared_set_sha256"]:
        raise ValueError("Boxplot shared-location membership differs from supplied visits")
    if audit["place_mixing"]["target_week"] != TARGET_WEEK or audit["place_mixing"]["minimum_n_rule"] != "n > 10":
        raise ValueError("Boxplot export must use week 9 and n > 10")
    if audit["host_travel"]["export_distance_unit"] != "km":
        raise ValueError("Boxplot distances must be in km")
    if [r["date"] for r in audit["host_travel"]["daily_counts"]] != expected_days:
        raise ValueError("Boxplot export must cover all seven target dates")
    expected_keys = {(m, g) for m in ["place_mixing", "host_travel_distance"] for g in GROUPS}
    if len(stats) != 4 or {(r["metric"], r["group"]) for r in stats} != expected_keys:
        raise ValueError("Expected four unique metric/group boxplot summaries")
    for row in stats:
        values = [row[k] for k in ["mean", "whislo", "q1", "median", "q3", "whishi"]]
        if row["n"] <= 0 or not np.isfinite(values).all():
            raise ValueError("Invalid boxplot sample or statistics")
        if not row["whislo"] <= row["q1"] <= row["median"] <= row["q3"] <= row["whishi"]:
            raise ValueError("Unordered boxplot quartiles or whiskers")
    return stats, audit


def compute_boxplots(place_file, daily_directory, host_file, shared, reference, unit, chunksize):
    daily = [Path(daily_directory) / f"{d:%Y-%m-%d}.csv" for d in pd.date_range(TARGET_START, TARGET_END)]
    for path in [Path(place_file), Path(host_file), *daily]:
        if not path.is_file(): raise FileNotFoundError(path)
    mixing, mixing_audit = mixing_values(place_file, shared, TARGET_WEEK, MIN_PLACE_N)
    travel, travel_audit = travel_values(daily, host_file, shared, chunksize, unit)
    stats = [boxplot_summary(values[group], metric, group, measurement_unit)
             for metric, values, measurement_unit in
             [("place_mixing", mixing, "index"), ("host_travel_distance", travel, "km")]
             for group in GROUPS]
    audit = dict(reference_start="2017-08-25", reference_end_inclusive="2017-09-07",
        target_start=TARGET_START, target_end_inclusive=TARGET_END,
        shared_threshold_distinct_people=3, shared_reference=reference,
        place_mixing=mixing_audit, host_travel=travel_audit,
        export_scope="Aggregate boxplot statistics only; no individual or POI rows")
    return validate_boxplots(stats, audit, reference)


def category_counts(host, shared, metadata_path, taxonomy_path):
    poi = pd.read_csv(metadata_path, dtype="string",
        usecols=["safegraph_place_id", "top_category", "sub_category"])
    taxonomy = pd.read_csv(taxonomy_path, dtype="string",
        usecols=["top_category", "sub_category", "Taxonomy"])
    mapped = poi.merge(taxonomy, on=["top_category", "sub_category"], how="inner", sort=False)
    mapped = mapped.rename(columns={"safegraph_place_id": "place_id"})[["place_id", "Taxonomy"]].dropna()
    duplicates = set(mapped.loc[mapped.place_id.duplicated(False), "place_id"])
    ambiguous = mapped.groupby("place_id").Taxonomy.nunique().gt(1)
    # Match the executed category analysis: first existing mapping per POI.
    mapped = mapped.drop_duplicates("place_id", keep="first")
    active = host.loc[host.week.eq(CATEGORY_WEEK), ["place_id"]].copy()
    active["shared_ge3"] = active.place_id.isin(shared)
    active = active.merge(mapped, on="place_id", how="left", validate="one_to_one")
    coverage = [dict(group="Shared" if flag else "Non-shared", n_host_active_pois=len(g),
        n_classified_pois=int(g.Taxonomy.notna().sum()), n_without_taxonomy=int(g.Taxonomy.isna().sum()))
        for flag, g in active.groupby("shared_ge3")]
    classified = active.dropna(subset=["Taxonomy"])
    counts = classified.groupby(["Taxonomy", "shared_ge3"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=[True, False], fill_value=0)
    counts.columns = ["shared_count", "nonshared_count"]
    if not (counts.sum() > 0).all(): raise ValueError("No classified POIs in one comparison group")
    counts["shared_proportion"] = counts.shared_count / counts.shared_count.sum()
    counts["nonshared_proportion"] = counts.nonshared_count / counts.nonshared_count.sum()
    audit = dict(category_week=CATEGORY_WEEK, category_dates="2017-09-01 to 2017-09-07",
        category_coverage=coverage, taxonomy_rule="First mapping per POI, matching executed analysis",
        n_active_pois_with_duplicate_mapping=len(set(active.place_id) & duplicates),
        n_active_pois_with_conflicting_categories=len(set(active.place_id) & set(ambiguous.index[ambiguous])),
        denominator="All taxonomy-classified host-active POIs in each group; one record per POI",
        displayed_categories=DISPLAY_CATEGORIES,
        undisplayed_categories=sorted(set(counts.index) - set(DISPLAY_CATEGORIES)),
        displayed_categories_renormalized=False)
    return counts, audit


def draw(stats, counts, output):
    plt = plotting()
    for metric, label, filename, ylim in [
        ("place_mixing", "Place-level social mixing", "social_mixing_boxplot", (0, 1.02)),
        ("host_travel_distance", "Host travel distance (km)", "host_travel_distance_boxplot", (0, 65))]:
        fig, ax = plt.subplots(figsize=(3.25, 3.8))
        rows = [next(s for s in stats if s["metric"] == metric and s["group"] == group) for group in GROUPS]
        boxes = ax.bxp([dict(label=s["group"], med=s["median"], q1=s["q1"], q3=s["q3"],
            whislo=s["whislo"], whishi=s["whishi"], fliers=[]) for s in rows],
            showfliers=False, patch_artist=True, widths=.48,
            medianprops={"color": "#303030", "linewidth": 1.4})
        for box, color in zip(boxes["boxes"], COLORS): box.set_facecolor(color)
        ax.set_ylabel(label); ax.set_ylim(*ylim)
        ax.set_xticks([1, 2], ["Shared", "Non-shared"])
        fig.subplots_adjust(left=.27, right=.96, bottom=.15, top=.96)
        save_plot(fig, output, filename); plt.close(fig)
    shown = counts.reindex(DISPLAY_CATEGORIES, fill_value=0)
    fig, ax = plt.subplots(figsize=(4.8, 6.4))
    y = np.arange(len(shown))
    ax.barh(y - .18, shown.shared_proportion, height=.36, color=COLORS[0], label="Shared locations")
    ax.barh(y + .18, shown.nonshared_proportion, height=.36, color=COLORS[1], label="Non-shared locations")
    ax.set_yticks(y, shown.index); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Proportion of POIs")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(.5, 1.015), fontsize=10)
    fig.subplots_adjust(left=.32, right=.97, bottom=.11, top=.87)
    save_plot(fig, output, "poi_taxonomy_comparison"); plt.close(fig)


def run(args):
    profile, shared, reference = build_reference(args.evacuee_visits)
    host = read_host_activity(args.host_activity)
    if args.boxplot_export is not None:
        export = Path(args.boxplot_export)
        stats = json.loads((export / "fig4d_boxplot_statistics.json").read_text())
        audit = json.loads((export / "audit.json").read_text())
        stats, box_audit = validate_boxplots(stats, audit, reference)
        # Keep the scientific audit without republishing original private paths.
        box_audit = {k: audit[k] for k in ["reference_start", "reference_end_inclusive",
            "target_start", "target_end_inclusive", "shared_threshold_distinct_people",
            "shared_reference", "place_mixing", "host_travel"]}
        stage = "Previously computed aggregate server boxplots, metadata validated"
    else:
        stats, box_audit = compute_boxplots(args.place_change, args.daily_visits, args.hosts,
            shared, reference, args.distance_unit, args.chunksize)
        stage = "Original place-week values and seven daily host-visit files"
    counts, categories = category_counts(host, shared, args.poi_metadata, args.taxonomy)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    counts.to_csv(output / "poi_taxonomy_comparison.csv")
    (output / "fig4d_boxplot_statistics.json").write_text(json.dumps(stats, indent=2) + "\n")
    (output / "audit.json").write_text(json.dumps(box_audit, indent=2) + "\n")
    (output / "location_comparison_audit.json").write_text(json.dumps(
        dict(shared_reference=reference, boxplot_input_stage=stage, **categories), indent=2) + "\n")
    if not args.no_figures: draw(stats, counts, output)
    return stats, counts, categories


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for option in ["evacuee-visits", "host-activity", "poi-metadata", "taxonomy", "output"]:
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--boxplot-export", type=Path)
    parser.add_argument("--place-change", type=Path)
    parser.add_argument("--daily-visits", type=Path)
    parser.add_argument("--hosts", type=Path)
    parser.add_argument("--distance-unit", choices=["m", "km"], default="m")
    parser.add_argument("--chunksize", type=int, default=500000)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    raw = [args.place_change, args.daily_visits, args.hosts]
    if args.boxplot_export is not None and any(x is not None for x in raw):
        parser.error("Choose --boxplot-export OR the three original-input options")
    if args.boxplot_export is None and not all(x is not None for x in raw):
        parser.error("Supply --boxplot-export, or --place-change --daily-visits --hosts")
    if args.chunksize <= 0: parser.error("--chunksize must be positive")
    run(args)
    print(f"Saved aggregate boxplots, categories and requested figures to {args.output}")


if __name__ == "__main__":
    main()
