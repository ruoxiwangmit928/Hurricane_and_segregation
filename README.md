# Hurricane Evacuation and Income Social Mixing

This repository contains the analysis code for the paper:

> **Divergence between residential and experienced social mixing during hurricane displacement**  
> [Authors] · [Journal/Conference] · [Year]

We examine how hurricane evacuations reshape income-based social mixing in affected communities, using large-scale mobility data from Hurricane Harvey (Houston, TX, 2017) and Hurricane Ian (Florida, 2022). The analysis covers three main components: (1) evacuation identification, (2) place- and individual-level social mixing computation, and (3) hypothesis testing on host population behavioral responses.

---

## Repository Structure

```
.
├── 1_evacuation_identification/
│   └── evacuation_identification.py
│
├── 2_social_mixing/
│   ├── place_social_mixing.py
│   └── individual_social_mixing.py
│
├── 3_host_population/
│   ├── host_population_counterfactual.py
│   └── tendency_to_explore.py
│
├── 4_hypothesis_testing/
│   ├── 1_encounter_index.py
│   └── 2_social_mixing_shared_places.py
│
├── data/                  # input data (not included — see Data section)
├── results/               # generated outputs
└── README.md
```

---

## Module Overview

### 1 · Evacuation Identification
**`evacuation_identification.py`**  
Identifies evacuation episodes from daily GPS displacement traces using a rolling threshold. An individual is classified as evacuated if their minimum distance from home exceeds `DISPLACEMENT_THRESHOLD` (default 1000 m) for ≥ 3 consecutive days, and as returned when they remain near home for ≥ 3 consecutive days. The pre-event window (Aug 10–20) is used to confirm baseline home location. Outputs a daily displacement status file and an episode-level table (start date, end date, duration).

### 2 · Social Mixing Computation

**`place_social_mixing.py`**  
Computes weekly place-level income social mixing for all visited POIs (Florida, Hurricane Ian 2022). Visitors are weighted by dwell time; social mixing is defined as:

$$\text{Mixing} = 1 - \frac{2}{3} \sum_{q \in \{Q1,Q2,Q3,Q4\}} \left| s_q - 0.25 \right|$$

where $s_q$ is the share of total dwell time attributable to income quartile $q$. A value of 1 indicates perfect mixing; 0 indicates complete segregation.

**`individual_social_mixing.py`**  
Computes the individual-level counterpart: each person's social mixing score is the dwell-time-weighted average of the place-level income exposure across all their visited POIs in a given week. Only active users (≥ 10 visit days per month) are included.

### 3 · Host Population Analysis

**`host_population_counterfactual.py`**  
Identifies the host population — residents who remained in the top 100 most-visited CBGs during the hurricane. Constructs counterfactual mobility baselines at three levels of stratification by randomly subsampling pre-event visits to match observed weekly dwell time: (1) total dwell time only; (2) stratified by travel-distance bin × income quartile; (3) further stratified by POI taxonomy category. Each level controls for progressively more confounders to isolate behavioral change from composition effects.

**`tendency_to_explore.py`**  
Measures the exploration tendency of the host population using the ratio of unique places visited to total visits ($p = S_t / N$). Tracks weekly changes in exploration and travel distance relative to a pre-hurricane baseline (weeks 0–5), and examines whether the post-event change correlates with CBG-level median income.

### 4 · Hypothesis Testing

**`hypothesis_1_encounter_index.py`**  
Tests whether evacuees and the host population co-visited the same POIs during the hurricane period. Computes two metrics weekly: an *Encounter Index* (dwell-time-weighted co-presence at shared POIs) and a *Host Shared Ratio* (fraction of host dwell time at evacuee-visited places). Also compares dwell-time change rates at evacuee-shared vs non-shared POIs.

**`hypothesis_2_social_mixing_shared_places.py`**  
Tests whether places visited by evacuees exhibit higher income social mixing than non-visited places. Produces a snapshot boxplot for the hurricane week, a weekly trend comparison, and a POI taxonomy breakdown of shared vs non-shared places.

---

## Social Mixing Index

The social mixing index used throughout is adapted from the dissimilarity index:

$$\text{Mixing} = 1 - \frac{2}{3} \sum_{q=1}^{4} \left| s_q - 0.25 \right|$$

| Value | Interpretation |
|-------|----------------|
| 1.0   | Perfect mixing — all income groups equally represented |
| 0.0   | Complete segregation — only one income group present |

Income quartiles (Q1–Q4) are assigned at the Census Block Group (CBG) level based on ACS median household income.

---

## Data

This study uses **Cuebiq** GPS mobility data under a Data for Good research agreement. The raw mobility data cannot be shared publicly. Intermediate aggregated outputs (e.g., weekly social mixing CSVs) may be available upon reasonable request.

| File | Description |
|------|-------------|
| `individual_minimum_distance.txt` | Daily minimum distance from home per device |
| `device_id_cbg_10_days.csv` | Home CBG assignment per device |
| `displacement_status.csv` | Daily displacement status (output of Step 1) |
| `displacement_periods.csv` | Evacuation episode table (output of Step 1) |
| `evacuees_poi_visits.csv` | POI visit records for identified evacuees |
| `match_results_remove_home_30/` | Daily POI visit files for all users |
| `mobility_<week>.txt` | Weekly mobility summaries (rg, distance, visits, dwell time) |
| `cbg_statistics.csv` | Evacuee inflow counts per CBG |
| `fl_cbg_income_quantile.csv` | Income quartile assignment per device (Florida) |
| `geo_change_features.gpkg` | CBG-level demographic and mobility features |
| `GH_poi_list.csv` / `taxonomy.csv` | POI metadata and category taxonomy |

---

## Requirements

```
python >= 3.8
pandas
numpy
geopandas
matplotlib
seaborn
scipy
statsmodels
haversine
mapclassify
```

Install dependencies:

```bash
pip install pandas numpy geopandas matplotlib seaborn scipy statsmodels haversine mapclassify
```

The scripts in `2_social_mixing/` additionally require a **Snowflake** connection via `cuebiq` internal tooling. Update the database paths in the configuration block before running:

```python
TABLE              = "YOUR_DATABASE.YOUR_SCHEMA.FL_ALL_VISITS_TARGET_PLACES"
ACTIVE_USERS_TABLE = "YOUR_DATABASE.YOUR_SCHEMA.FL_ACTIVE_USERS"
```

All other scripts read from local files. Update the path constants at the top of each file to match your local directory layout.

---

## Module Summary

| Folder | Script | Description |
|--------|--------|-------------|
| `1_evacuation_identification` | `evacuation_identification.py` | Identifies evacuees from GPS traces using a displacement threshold |
| `2_social_mixing` | `place_social_mixing.py` | Weekly place-level social mixing (Florida / Hurricane Ian) |
| `2_social_mixing` | `individual_social_mixing.py` | Weekly individual-level social mixing for active users |
| `3_host_population` | `host_population_counterfactual.py` | Identifies host population and builds a counterfactual dwell-time baseline |
| `3_host_population` | `tendency_to_explore.py` | Tracks weekly exploration and dwell-time changes for the host population |
| `4_hypothesis_testing` | `1_encounter_index.py` | Computes evacuee–host co-presence metrics at shared POIs |
| `4_hypothesis_testing` | `2_social_mixing_shared_places.py` | Compares social mixing at evacuee-visited vs non-visited places |

---

## Citation

If you use this code, please cite:

```bibtex
@article{[citekey],
  title   = {[Paper Title]},
  author  = {[Authors]},
  journal = {[Journal]},
  year    = {[Year]},
  doi     = {[DOI]}
}
```

---

## License

This repository is released under the [MIT License](LICENSE).
