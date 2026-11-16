# Supplementary analyses

Principal analysis programs for the hurricane displacement and social-mixing study. The structure follows research questions, not figure numbers. Each program accepts explicit inputs and an output directory. No notebooks, historical plotting variants, manuscript-building programs or data files are included.

## Analysis modules

| Topic | Program | Analysis |
|---|---|---|
| Displacement identification | `identification/harvey_window_sensitivity.py` | Onset-window shifts, destination validation and receiving-area stability from stored spells and nighttime observations |
| Displacement identification | `identification/ian_window_sensitivity.py` | Daily away-observation thresholds, onset-window shifts and destination stability on the authorized platform |
| Measurement robustness | `measurement_robustness/harvey_income_robustness.py` | Alternative socioeconomic assignments, matched individual agreement and lower within-neighborhood income variation |
| Measurement robustness | `measurement_robustness/ian_income_robustness.py` | Income-assignment sensitivity with normalization within each draw before averaging |
| Shared locations | `shared_locations/harvey_shared_location_construction.py` | Fixed acute/extended reference profiles, distinct-visitor thresholds and weekly host exposure summaries |
| Shared locations | `shared_locations/harvey_shared_location_sensitivity.py` | Sensitivity of dwell time, Encounter Index and Host Shared Ratio to the reference period and visitor threshold |
| Host activity and recovery | `host_recovery/harvey_demand_pressure.py` | Host recovery associations before/after adjustment for measured demand pressure; absorbed location/week effects and cluster inference |
| Host activity and recovery | `host_recovery/ian_recovery_patterns.py` | Aggregate mixing recovery, location-selection sensitivity and optional dwell/overlap trajectories |
| Community context | `community_context/receiving_counts.py` | Metropolitan and original Top-100 receiving-count models, environmental/socioeconomic block tests and income-adjustment comparisons |
| Community context | `community_context/income_environment_adjustment.py` | Original Top-100 cohort, environmental adjustment of income associations and explicit complete-case sample accounting |
| Community context | `community_context/ian_community_heterogeneity.py` | Local recovery differences, community characteristics and host/origin income comparisons |

`common/ian_metrics.py` is a shared dependency for the three Ian measurement, recovery and community programs. Include it whenever any of those programs is used.

## Running a program

Use Python 3.10+ and install the relevant packages in `requirements.txt`. Consult the script's opening docstring and CLI help for input columns and options:

```bash
python measurement_robustness/ian_income_robustness.py --help
python community_context/receiving_counts.py --help
python community_context/income_environment_adjustment.py --help
```

Example:

```bash
python community_context/income_environment_adjustment.py \
  --roster /path/to/original_top100.csv \
  --outcomes /path/to/current_cbg_outcomes.csv \
  --covariates /path/to/cbg_environment.csv \
  --output /path/to/results/income_environment \
  --plots
```

The Top-100 roster defines the cohort. Left joins retain its 100 members before an explicit common complete-case screen. The program records the observed analysis sample size, missing fields and excluded communities automatically. The roster size is not substituted for the number of observations actually fitted.

## Data stages

- **Identification:** the Harvey program starts from stored displacement spells, pre-event homes and nighttime stays. It does not recreate spells from GPS pings. The Ian program starts from an existing standardized daily home-trajectory table and original candidate table. It accepts an authorized engine through `run(...)` or `--engine-factory module:function`; its four derived SQL output tables use an explicit `--result-prefix`. It does not initialize platform access or store credentials.
- **Income robustness:** Harvey starts from the saved fixed-support individual estimates; Ian starts from saved weekly estimates for each assignment. These programs analyze existing assignments, rather than recreating the upstream income-imputation pipeline.
- **Shared-location construction:** requires prepared evacuee visit rows covering August 25–September 28 and all twelve weekly host–location dwell tables. The sensitivity program consumes the six aggregate CSVs produced by that construction. It can also read the existing six-table export.
- **Demand pressure:** requires the saved, balanced, baseline-selected location-week panel with verified displacement/return timing. It does not recreate the original host cohort or daily visit panel.
- **Community models:** require prepared CBG covariates, original sample membership and/or saved CBG-week summaries. The programs never infer missing outcomes from figures or fill them with invented data.

## Measures and samples

Mixing changes use `100*(M-M0)/(1-M0)`, with `M=1-S`, where the current analysis calls for normalized mixing change. Absolute-level income agreement remains on M. Dwell time, Encounter Index and Host Shared Ratio keep their own metric-specific baseline changes.

The Ian local-recovery comparison uses the Top-100 host subset of the original Top-200 measurement universe. Fixed location profiles and visitor composition are retained, yielding the existing 12/88 groups. Receiving-count regressions use distinct observed arrivals; the separate environmental-adjustment analysis retains the original peak-net population outcome. These are distinct analyses.

Internal input column names are preserved to remain compatible with the existing data exports. Analysis titles and filenames describe the research task rather than historical reviewer or figure labels.

## Verification and scope

The refactored Ian tables, community-context models and demand-pressure results were checked against saved analyses. Shared-location sensitivity retains the numerical consistency checks. CLI and syntax checks cover the principal programs. Platform jobs, complete raw-trajectory reconstruction and new income draws were not rerun during preparation; this is a set of research analysis entry points, not an unqualified one-command raw-data pipeline.

Input data and generated private intermediates are not distributed with this code. Keep outputs outside the code repository. Cuebiq data access remains subject to the provider's conditions.
