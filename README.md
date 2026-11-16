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

## Shared locations and the fixed reference

For the Harvey shared/non-shared comparison, count distinct identified evacuees at each location during **August 25–September 7, 2017, inclusive**, using visits inside each person's recorded displacement episode. A shared location has at least **three distinct evacuees**. Repeated visits by one person count once. Non-shared locations may therefore have zero, one, or two evacuees. The classification is fixed across all host-observation weeks.

The approved main-figure calculations distinguish two reference sets:

- **Complete-reference overlap (panel b):** both the Encounter Index and Host Shared Ratio retain the complete acute-period evacuee location profile. No three-person filter is imposed on this profile. Weekly host activity uses the original Top-100 host sample.
- **Shared/non-shared comparisons (panels c–e):** use the three-person classification. Weekly dwell time is the mean total host dwell time per host-active location, with baseline-relative changes using weeks 0–5. The mixing/travel snapshot uses week 9 (September 8–14), the stored `n > 10` mixing filter, absolute M, and travel distances in km. Location-category composition uses week 8 (September 1–7), counts each location once, and retains all classified categories in the denominator.

In the SI, thresholds 1, 3, 5, and 10 and an extended reference period are sensitivity specifications. 

## Data access and running code

The mobility data were provided by Cuebiq under licence and are not publicly distributed. Use authorized local files or database tables and write generated outputs outside the source tree. No access credentials are supplied.

Use Python 3.10 or newer for the revised entry points. Core local analyses require `numpy`, `pandas`, `matplotlib`, `scipy`, `statsmodels`, and `patsy`; original geographic scripts additionally use packages such as `geopandas`, `haversine`, and `mapclassify`. See [SI requirements](supplementary_analysis/requirements.txt). Original plotting templates may require `seaborn`.

Platform entry points accept a user-supplied authorized engine; they do not establish data access. The original absolute-index templates require an initialized `snow_engine` and configured table paths. Keep observed values, sample exclusions, reference metadata, and aggregate audits together when regenerating results.
