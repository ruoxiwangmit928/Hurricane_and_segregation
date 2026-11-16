# Supplementary analyses

This directory contains the principal supplementary analyses for the study of hurricane displacement and income-based social mixing. The analyses assess how the findings depend on displacement identification, socioeconomic measurement, and the definition of shared locations. They also examine the relationship between evacuee arrivals, host activity, and neighborhood characteristics, and compare recovery patterns across receiving communities.

The code is organized by analytical question. The sections below describe what each analysis evaluates, how its measures are constructed, and how its results relate to the main study.

## Social-mixing measures

Experienced social mixing is denoted by M, with M = 1 − S, where S is the corresponding segregation index. Absolute mixing values describe the balance of income-group exposure. Temporal changes are expressed as:

$$\Delta M_t = 100\frac{M_t-M_0}{1-M_0},$$

where M₀ is the pre-event mixing mean. Negative values indicate reduced mixing. This normalization uses the complement of baseline mixing, rather than baseline mixing itself. Harvey normalized trajectories use weeks 0–5 as the baseline; Ian trajectories use the seven weeks beginning August 5 through September 16, 2022.

Absolute mixing remains the outcome for comparisons of individual estimates across socioeconomic assignments. Community analyses that report the magnitude of a decline use −ΔM. Dwell time, the Encounter Index, and the Host Shared Ratio retain their own baseline-relative percentage changes. These activity-based measures characterize potential exposure and spatial overlap, rather than directly observed interpersonal interactions.

## 1. Displacement identification and destination stability

**Harvey — [harvey_window_sensitivity.py](identification/harvey_window_sensitivity.py)**

This analysis evaluates the sensitivity of the identified evacuee population and major receiving neighborhoods to the boundaries of the displacement-onset window. It varies the start and end dates around the primary August 24–September 2, 2017 window, using recorded displacement episodes, pre-event home locations, and nighttime observations to apply destination-validation criteria. Comparisons of individual membership and Top-100 receiving-area membership distinguish changes in the identified population from changes in the geography of reception.

**Ian — [ian_window_sensitivity.py](identification/ian_window_sensitivity.py)**

The Ian analysis examines both onset-window boundaries and the number of successive observed away records required to identify sustained absence from home. It compares these specifications with the primary September 23–October 10, 2022 window and assesses the stability of identified individuals and receiving destinations. The calculations use prepared daily trajectories; successive observed records need not represent consecutive calendar days.

## 2. Socioeconomic measurement and robustness

**Harvey — [harvey_income_robustness.py](measurement_robustness/harvey_income_robustness.py)**

This analysis compares the original home-CBG median-income assignment with alternative income quantiles, mean-income assignments, and distribution-based assignments derived from within-CBG income bins. Matched person–week observations allow differences in the mixing estimates to be examined without changing the observation sample across the unrestricted specifications. Pearson and Spearman correlations summarize agreement in absolute individual mixing, while weekly trajectories assess whether the event-period decline and subsequent recovery depend on the income assignment.

Distribution-based estimates are pooled across 20 assignments within each person–week before weekly aggregation. Both absolute mixing and normalized temporal changes are reported. A complementary analysis restricts the sample to home CBGs whose estimated internal income variation is at or below the sample median, addressing the sensitivity of the results to the use of a neighborhood-level socioeconomic proxy.

**Ian — [ian_income_robustness.py](measurement_robustness/ian_income_robustness.py)**

For hosts in the Top-100 receiving sample, this analysis compares the median-income specification with uniform-within-bin and log-normal assignments. Each alternative includes 100 assignments. Normalized changes are calculated within each assignment using its own pre-event baseline and are then averaged across assignments. The comparison retains the matched weekly observation support and reports both mixing levels and temporal changes, distinguishing differences in absolute values from differences in the recovery pattern.

## 3. Shared-location construction and definition sensitivity

**Construction — [harvey_shared_location_construction.py](shared_locations/harvey_shared_location_construction.py)**

Shared locations are defined using visits made by identified evacuees within their recorded displacement episodes. The primary Harvey reference covers August 25–September 7, 2017, including both dates, and classifies a location as shared when it was visited by at least three distinct evacuees. Repeated visits by the same person do not increase this count. Non-shared locations may therefore include locations visited by one or two evacuees. The location classification and evacuee dwell-time profile are constructed once and held fixed across weekly host observations.

**Sensitivity — [harvey_shared_location_sensitivity.py](shared_locations/harvey_shared_location_sensitivity.py)**

The sensitivity analysis compares thresholds of 1, 3, 5, and 10 distinct evacuees and extends the fixed reference through September 28. It quantifies changes in location membership and in host dwell time, the Encounter Index, and the Host Shared Ratio. Host dwell time is measured as mean total host dwell time per active location within each group and week. The Encounter Index describes the alignment of weekly host dwell time with the fixed evacuee profile; the Host Shared Ratio describes the fraction of host dwell time at reference locations.

For these sensitivity specifications, the evacuee profile is restricted and renormalized at each threshold. This evaluates how the definition affects the measured trajectories. It is distinct from the complete-reference overlap calculation retained in main-text Figure 4b; the shared/non-shared comparisons in panels c–e use the three-person definition.

## 4. Host activity, demand pressure, and recovery

**Harvey demand pressure — [harvey_demand_pressure.py](host_recovery/harvey_demand_pressure.py)**

This analysis examines whether the association between acute-period evacuee exposure and subsequent host activity changes after accounting for measured local demand pressure. Descriptive trajectories cover the pre-event, acute-displacement, and recovery periods. The recovery models compare weeks 0–5 with weeks 9–11 in a baseline-selected location–week panel, including location and week fixed effects with uncertainty clustered by location.

The demand proxy combines baseline-expected host visits with observed visits by evacuees and other users, scaled by the location's pre-event all-user visitation benchmark (the 90th percentile). Using baseline-expected host visits avoids inserting the observed host outcome directly into the host-demand component. The analysis assesses an observable demand-based explanation for reduced activity; it does not measure service capacity or identify residents' motivations.

**Ian recovery — [ian_recovery_patterns.py](host_recovery/ian_recovery_patterns.py)**

This analysis compares normalized experienced-mixing trajectories for hosts, evacuees, and other visitors in the Top-100 receiving-area analysis. It also evaluates host-mixing sensitivity to the fixed-location visitor threshold and summarizes shared/non-shared activity and overlap trajectories. The combination of measures distinguishes the initial disruption from the subsequent pace of aggregate recovery.

## 5. Receiving-neighborhood characteristics and local heterogeneity

**Receiving-count models — [receiving_counts.py](community_context/receiving_counts.py)**

These models examine how the number of distinct qualifying evacuees received by a CBG is associated with environmental conditions, socioeconomic composition, and accessibility. Poisson pseudo-maximum-likelihood models with a log link include common size and observation-coverage controls, county fixed effects, and census-tract-clustered inference. Environmental and socioeconomic characteristics are assessed separately and jointly; extensions add road density and hospital distance, or a supplementary FEMA monetary indicator.

Models are estimated separately for the metropolitan sample and eligible members of the original Top-100 receiving sample. Joint tests assess the environmental and socioeconomic variable blocks, while additional comparisons track the income coefficient as controls are introduced. Model-specific complete-case samples are recorded. These estimates describe conditional receiving-count associations rather than identifying a destination-choice mechanism.

**Environmental adjustment of income relationships — [income_environment_adjustment.py](community_context/income_environment_adjustment.py)**

This analysis examines whether the descriptive income relationships change after adjustment for measured environmental characteristics. The three outcomes are peak net population increase, the magnitude of normalized host-mixing decline, and reduced place exploration. The inflow measure here is peak net increase, distinct from the qualifying-arrival counts used in the receiving-count models.

Nested linear models add coast/bay distance, mapped flood-zone shares, and a supplementary FEMA monetary indicator, with HC3 heteroskedasticity-robust inference. Pairwise Spearman correlations also describe the overlap between income and the environmental indicators. All outcome/model comparisons use one common complete-case sample drawn from the original Top-100 roster. Available information determines the fitted sample, and exclusions are recorded rather than changing the original roster definition.

**Ian community heterogeneity — [ian_community_heterogeneity.py](community_context/ian_community_heterogeneity.py)**

This analysis evaluates whether aggregate recovery conceals more prolonged changes in particular receiving communities. It retains the exposure measurements and visitor composition established in the Top-200 analysis and examines the Top-100 receiving subset. The joint response screen considers experienced mixing, the Encounter Index, and the Host Shared Ratio, identifying 12 communities for comparison with the other 88.

Mixing changes and baseline-relative activity changes are calculated within each CBG before calculating equally weighted group means and medians. Comparisons also describe receiving intensity, neighborhood and origin income, host–origin income differences, and observed displacement duration. Because the groups are defined using their observed responses, these comparisons characterize local heterogeneity rather than providing an independent test of its causes.

## Analytical scope and data

The analyses use prepared visit records, individual estimates, location–week panels, and CBG-level summaries described in the Methods and Supplementary Information. They assess measurement sensitivity and observed associations at complementary spatial and temporal scales. The shared Ian definitions and aggregation functions are collected in [ian_metrics.py](common/ian_metrics.py).

Mobility data are subject to Cuebiq's licensing and access conditions and are not included in this repository. The public code documents the supplementary calculations without distributing individual mobility records.
