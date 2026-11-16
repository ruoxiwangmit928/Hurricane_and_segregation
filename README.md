# Hurricane displacement and income-based social mixing

Analysis code for *Divergence between residential and experienced social mixing during hurricane displacement*, covering Hurricane Harvey (2017) and Hurricane Ian (2022).

Programs start from the data stages described in their docstrings. Raw mobility data and private intermediate tables are not distributed. This repository does not provide a single raw-GPS-to-final-results command.

## Analysis structure

| Directory | Purpose |
|---|---|
| `1_evacuation_identification/` | Original displacement-identification code |
| `2_social_mixing/` | Absolute mixing measures, current Ian aggregate calculation, and downstream normalized changes |
| `3_host_population/` | Original simulation visit generation and place-exploration analysis |
| `4_hypothesis_testing/` | Fixed-reference overlap, host activity, and shared/non-shared location comparisons |
| `supplementary_analysis/` | Identification sensitivity, measurement robustness, shared-location sensitivity, recovery, and community context |

See [the supplementary analysis guide](supplementary_analysis/README.md) for its topic-based program list. Run current command-line entry points with `--help` for arguments and input schemas. Original scripts requiring an existing database connection are explicitly identified as such.

## Mixing levels and normalized changes

For four income groups, place-level mixing is

$$M_{\alpha}=1-\frac{2}{3}\sum_q\left|\tau_{q\alpha}-\frac14\right|,$$

where $\tau_{q\alpha}$ is the income group's share of total dwell time at that location. Individual exposure first combines the income shares of visited locations using the individual's dwell-time weights:

$$\tau_{iq}=\sum_{\alpha}\tau_{i\alpha}\tau_{q\alpha}, \qquad
M_i=1-\frac{2}{3}\sum_q\left|\tau_{iq}-\frac14\right|.$$

Both are absolute indices, $M=1-S$, ranging from 0 to 1. Individual mixing is calculated from weighted exposure shares, not by averaging place mixing indices. These indices describe potential socioeconomic exposure, not directly observed interpersonal interaction.

Time trajectories expressed as **normalized mixing change** use

$$\Delta M_t=100\frac{M_t-M_0}{1-M_0}
=-100\frac{S_t-S_0}{S_0}.$$

Negative values indicate reduced mixing. This is not percentage change relative to baseline $M$. A positive decline outcome, used in the community-income comparison, is $-\Delta M$.


## Current Ian calculation and original templates

`2_social_mixing/ian_social_mixing.py` is extracted from the executed threshold-sensitivity calculation that generated the current aggregate Ian input. It starts from existing bounded visitation/income tables and prepared cohort labels. The analysis covers twelve weeks beginning August 5, 2022 and ending October 27, with seven pre-event weeks and a fixed September 23–October 10 reference. Its configuration and required upstream tables are documented in the program. An authorized database connection is required.

This aggregate calculation and the separately prepared GE3EVAC inputs used for Ian income-assignment robustness are distinct stages. The SI income program continues to consume the existing assignment-specific exports; the new aggregate entry point does not recreate those assignments.

`place_social_mixing.py` and `individual_social_mixing.py` retain the original absolute-index templates and their original cohort logic. In particular, the original individual template starts on August 19 and uses a monthly activity filter. The current Ian calculation has different eligibility and location filters; changing the original start date alone would not reproduce it. The templates are retained for provenance and should not be presented as the current Ian cohort generator.

Saved current Ian results can be processed locally by the SI programs for recovery, socioeconomic robustness, and community heterogeneity. The original simulation script in `3_host_population/` generates visit samples; it does not by itself calculate or render the final simulated mixing series.

## Shared locations and the fixed reference

For the Harvey shared/non-shared comparison, count distinct identified evacuees at each location during **August 25–September 7, 2017, inclusive**, using visits inside each person's recorded displacement episode. A shared location has at least **three distinct evacuees**. Repeated visits by one person count once. Non-shared locations may therefore have zero, one, or two evacuees. The classification is fixed across all host-observation weeks.

The approved main-figure calculations distinguish two reference sets:

- **Complete-reference overlap (panel b):** both the Encounter Index and Host Shared Ratio retain the complete acute-period evacuee location profile. No three-person filter is imposed on this profile. Weekly host activity uses the original Top-100 host sample.
- **Shared/non-shared comparisons (panels c–e):** use the three-person classification. Weekly dwell time is the mean total host dwell time per host-active location, with baseline-relative changes using weeks 0–5. The mixing/travel snapshot uses week 9 (September 8–14), the stored `n > 10` mixing filter, absolute M, and travel distances in km. Location-category composition uses week 8 (September 1–7), counts each location once, and retains all classified categories in the denominator.

`4_hypothesis_testing/1_encounter_index.py` and `2_social_mixing_shared_places.py` implement these distinct calculations; `shared_location_reference.py` supplies the common episode/window/distinct-visitor construction. Consult their CLI help for prepared inputs and server-export workflows. Keep the helper alongside both scripts.

In the SI, thresholds 1, 3, 5, and 10 and an extended reference period are sensitivity specifications. There, the Encounter Index profile is restricted and renormalized at each threshold. Those series are not interchangeable with complete-reference main panel b. Figure captions must retain this distinction.

## Data access and running code

The mobility data were provided by Cuebiq under licence and are not publicly distributed. Use authorized local files or database tables and write generated outputs outside the source tree. No access credentials are supplied.

Use Python 3.10 or newer for the revised entry points. Core local analyses require `numpy`, `pandas`, `matplotlib`, `scipy`, `statsmodels`, and `patsy`; original geographic scripts additionally use packages such as `geopandas`, `haversine`, and `mapclassify`. See [SI requirements](supplementary_analysis/requirements.txt). Original plotting templates may require `seaborn`.

Platform entry points accept a user-supplied authorized engine; they do not establish data access. The original absolute-index templates require an initialized `snow_engine` and configured table paths. Keep observed values, sample exclusions, reference metadata, and aggregate audits together when regenerating results.
