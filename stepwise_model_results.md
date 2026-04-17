# Stepwise Variable-by-Variable Models — WPS Enforcement Analysis
**Zimmerman/HLM Approach | Keshav Goel | April 2026**

---

## Overview

This document reports results from the variable-by-variable model-building sequence following the Zimmerman (2000) / Raudenbush & Bryk (2002) HLM approach. Each predictor is entered one at a time into the unconditional model, and the reduction in between-state variance is tracked at each step.

**Sample:** 250 observations, 47 states, 2011–2019  
**Outcome:** WPS violations per state-year  
**Random effects:** Random intercept + random slope for time, by state (REML)  
**Time variable:** Centered on 2017 (time = year − 2017)

---

## Team Analytical Decisions (from email thread)

The following decisions were reached before running these models and are reflected in the variable sequence below.

**Spending operationalization: dollars per WPS-regulated agricultural establishment**

Stephane's rationale, agreed to by Joe and Kaitlyn:
- The spending variable needs to answer: *does the state have sufficient resources to cover its inspection workload?*
- Inspection workload is defined by the number of **establishments an inspector must visit**, not by the number of workers employed there
- Per-establishment funding directly matches the numerator (dollars allocated to enforcement) to the denominator (regulated units that enforcement must cover)
- Aim 3 already includes agricultural workforce size as a standalone predictor; using dollars per 1,000 workers would put the spending denominator and a separate predictor on the same underlying measure — this creates structural variance overlap, requiring VIF management and risking coefficient suppression
- Per-establishment spending does not share this structural overlap

**Land mass as a covariate** (Joe's addition, agreed by Kaitlyn):
- Visiting 100 establishments within 100 square miles is substantially easier than 100 establishments spread across 10,000 square miles
- State land area (sq miles) included as a fixed covariate

**Covariates from Yuri's data:**
- Total farming operations by state (Census of Agriculture) and H-2A workers per operation are available for the variable-by-variable sequence; added one predictor at a time per the Zimmerman approach

---

## ECHO Establishment Coverage Audit

Stephane requested confirmation that EPA ECHO WPS establishment counts are available and consistent across all 50 states for **2015–2022** before committing to per-establishment as the spending denominator. Findings below.

**Current ECHO data on hand covers 2014–2019 only.** Years 2020–2022 are not yet in the project files — this needs to be pulled before per-establishment spending can be used for any analysis extending beyond 2019.

| Year | Coverage | Notes |
|------|----------|-------|
| 2011 | **Not available** | Outside ECHO data on hand |
| 2012 | **Not available** | Outside ECHO data on hand |
| 2013 | **Not available** | Outside ECHO data on hand |
| 2014 | 48/50 states | CO and CT missing |
| 2015 | 50/50 states — complete | |
| 2016 | 50/50 states — complete | |
| 2017 | 50/50 states — complete | |
| 2018 | 50/50 states — complete | |
| 2019 | 50/50 states — complete | |
| 2020 | **Not yet pulled** | Needed if study period extends to 2022 |
| 2021 | **Not yet pulled** | Needed if study period extends to 2022 |
| 2022 | **Not yet pulled** | Needed if study period extends to 2022 |

**Action items:**
- Pull ECHO WPS establishment counts for 2020–2022 and re-audit before committing to per-establishment as the final denominator
- For **2011–2013**: substitute USDA Census of Agriculture farm counts (2012 and 2017 census values; interpolate non-census years). Must be documented in the methods section.
- For **CO and CT in 2014**: imputed from their 2015 values for this analysis (CO = 130, CT = 47)

---

## Model 0 — Unconditional Baseline

> **violations ~ time + time² + time³ + (1 + time | state)**

| Parameter | Estimate |
|-----------|----------|
| Intercept | 17.46 |
| time | 2.97*** |
| time² | −0.63† |
| time³ | −0.15** |
| **σ²_u0 (between-state)** | **325.71** |
| σ²_ε (within-state) | 95.99 |
| **ICC** | **0.772** |
| Log-Likelihood | −974.99 |

†p < .10, \*p < .05, \*\*p < .01, \*\*\*p < .001

**77.2% of total variance is between states**, confirming strong state-level clustering and justifying the HLM structure.

---

## Covariate Preparation Notes

All Level-2 predictors are **z-score standardized** (mean = 0, SD = 1) before entry so coefficients are directly comparable across predictors.

| Covariate | Source | Type | Mean (raw) | SD (raw) |
|-----------|--------|------|-----------|---------|
| `spending_per_operation` | Mean 2017$ spending / Census 2017 farming operations | State-level | $19.56 | $47.97 |
| `land_area_sqmi` | US Census Bureau (total sq miles) | State-level, constant | 75,891 | 97,066 |
| `spending_per_estab` | Mean 2017$ spending / ECHO establishment mean (2015–2019) | State-level | $5,209 | $13,949 |
| `operations` | Total farming operations, Census 2017 (from Yuri) | State-level | 40,844 | 39,314 |
| `h2a_workers` | H-2A workers per state, 2017 (from Yuri) | State-level | 3,340 | 6,562 |
| `workers_per_operation` | H-2A workers / total operations, 2017 (from Yuri) | State-level | 0.098 | 0.158 |

> **Note on dollars-per-acre:** The priority operationalization from Option 1 (spending / harvested cropland acres from USDA NASS Quick Stats) requires data not currently in the project files. `spending_per_operation` is used as the closest available proxy; Census of Agriculture operations are the recommended fallback per Stephane's email.

---

## Stepwise Model Results

For each predictor, two models are fit:
- **(a) Main effect only** — predictor entered as a fixed effect
- **(b) Main effect + linear time interaction** — tests whether the predictor moderates the linear time trend

The key metric is **% between-state variance explained** relative to the M0 baseline (σ²_u0 = 325.71).

---

### 1. Spending per Farming Operation

| Model | β (standardized) | p | σ²_u0 | Δσ²_u0 | % Explained | LogLik |
|-------|-----------------|---|-------|---------|-------------|--------|
| (a) Main only | −1.58 | .318 | 316.98 | +8.72 | 2.7% | −972.70 |
| (b) Main + ×time | −4.62 / ×time: −0.81 | .142 / .268 | 318.19 | +7.52 | 2.3% | −971.52 |

Neither coefficient is statistically significant. Spending scaled by farming operations explains roughly 2–3% of between-state variance.

---

### 2. State Land Area (sq miles)

| Model | β (standardized) | p | σ²_u0 | Δσ²_u0 | % Explained | LogLik |
|-------|-----------------|---|-------|---------|-------------|--------|
| (a) Main only | −0.80 | .581 | 325.21 | +0.49 | 0.2% | −973.79 |
| (b) Main + ×time | 0.02 / ×time: 0.19 | .994 / .764 | 328.68 | −2.97 | −0.9% | −973.35 |

Land area explains essentially none of the between-state variance and is not significant. Joe's travel-burden hypothesis is theoretically sound but not supported by this data.

---

### 3. Spending per ECHO Establishment

| Model | β (standardized) | p | σ²_u0 | Δσ²_u0 | % Explained | LogLik |
|-------|-----------------|---|-------|---------|-------------|--------|
| (a) Main only | −1.98 | .225 | 311.30 | +14.41 | 4.4% | −972.33 |
| (b) Main + ×time | −4.78 / ×time: −0.76 | .121 / .293 | 311.98 | +13.72 | 4.2% | −971.10 |

Slightly stronger than per-operation spending (~4% variance explained), but still not statistically significant. The negative direction is consistent with the expected direction — states with more funding per establishment have lower violations — but the effect is not reliably estimated with this sample.

---

### 4. Total Farming Operations (Census 2017)

| Model | β (standardized) | p | σ²_u0 | Δσ²_u0 | % Explained | LogLik |
|-------|-----------------|---|-------|---------|-------------|--------|
| (a) Main only | 0.98 | .609 | 271.46 | +54.25 | **16.7%** | −973.39 |
| (b) Main + ×time | 7.92 / ×time: 1.30 | **.003** / **.003** | 264.67 | +61.04 | **18.7%** | −969.25 |

The largest reduction in between-state variance of any single predictor. The time interaction is highly significant — states with more farming operations have a steeper positive trend in violations over time.

---

### 5. H-2A Workers per State (2017)

| Model | β (standardized) | p | σ²_u0 | Δσ²_u0 | % Explained | LogLik |
|-------|-----------------|---|-------|---------|-------------|--------|
| (a) Main only | 3.19 | **< .001** | 271.39 | +54.32 | **16.7%** | −967.61 |
| (b) Main + ×time | 9.94 / ×time: 1.39 | **.001** / **.023** | 266.10 | +59.61 | **18.3%** | −964.63 |

Nearly identical variance absorption as total farming operations. H-2A workers and total operations are likely highly correlated — **VIF should be checked before including both in the same model.**

---

### 6. H-2A Workers per Operation (2017)

| Model | β (standardized) | p | σ²_u0 | Δσ²_u0 | % Explained | LogLik |
|-------|-----------------|---|-------|---------|-------------|--------|
| (a) Main only | 2.32 | .046 | 329.78 | −4.07 | −1.3% | −974.06 |
| (b) Main + ×time | 5.04 / ×time: 0.54 | .114 / .378 | 327.43 | −1.73 | −0.5% | −969.44 |

Although the main effect reaches nominal significance (p = .046), between-state variance actually *increases* slightly relative to M0. This suggests instability — possibly driven by multicollinearity with `h2a_workers` or `operations`, or the ratio masking the underlying scale differences across states.

---

## Summary Variance Decomposition Table

| Model | Predictor | σ²_u0 | σ²_ε | Δσ²_u0 | % Explained | LogLik |
|-------|-----------|-------|------|---------|-------------|--------|
| M0 | Time only (baseline) | 325.71 | 95.99 | — | — | −974.99 |
| M1a | + spending/operation | 316.98 | 97.22 | +8.72 | 2.7% | −972.70 |
| M1b | + spending/operation × time | 318.19 | 96.88 | +7.52 | 2.3% | −971.52 |
| M2a | + land area | 325.21 | 96.33 | +0.49 | 0.2% | −973.79 |
| M2b | + land area × time | 328.68 | 96.36 | −2.97 | −0.9% | −973.35 |
| M3a | + spending/establishment | 311.30 | 97.56 | +14.41 | 4.4% | −972.33 |
| M3b | + spending/establishment × time | 311.98 | 97.46 | +13.72 | 4.2% | −971.10 |
| M4a | + farming operations | 271.46 | 101.60 | +54.25 | **16.7%** | −973.39 |
| M4b | + farming operations × time | 264.67 | 98.85 | +61.04 | **18.7%** | −969.25 |
| M5a | + H-2A workers | 271.39 | 96.61 | +54.32 | **16.7%** | −967.61 |
| M5b | + H-2A workers × time | 266.10 | 95.09 | +59.61 | **18.3%** | −964.63 |
| M6a | + workers/operation | 329.78 | 92.32 | −4.07 | −1.3% | −974.06 |
| M6b | + workers/operation × time | 327.43 | 95.00 | −1.73 | −0.5% | −969.44 |

---

## Key Takeaways

**1. Spending variables explain little between-state variance (~2–4%) and are not statistically significant.** This is worth discussing with the team — it may reflect that raw or normalized spending levels are not the primary driver of state-level baseline violations, or that the spending measure lacks sufficient variation once normalized.

**2. Total farming operations and H-2A workers are the strongest single predictors**, each absorbing ~17% of between-state variance. Both are also significant when interacted with time, suggesting that agricultural scale predicts not just baseline violation rates but the trajectory of violations over the study period.

**3. Farming operations and H-2A workers are likely collinear.** Including both in the same model risks coefficient suppression. VIF analysis is the next step before adding both together.

**4. Land area does not empirically reduce between-state variance.** While the travel-burden hypothesis is plausible, the data do not support it as a meaningful covariate in this model.

**5. Workers per operation adds noise rather than signal.** The ratio measure may be less stable than its constituent parts; the team should decide whether to retain it.

---

## Next Steps

- [ ] **Pull ECHO WPS establishment counts for 2020–2022** (Stephane's specific ask) — required before per-establishment spending can be used for any analysis period extending to 2022
- [ ] Obtain USDA Census of Agriculture farm counts (2012 and 2017) to fill the 2011–2013 ECHO establishment gap; document substitution in methods
- [ ] Obtain USDA NASS Quick Stats harvested cropland acres by state for the study period to compute the priority dollars-per-acre operationalization (Option 1, identified as the primary operationalization in the original email thread)
- [ ] Check VIF for farming operations + H-2A workers before including both in a joint model — they absorb nearly identical between-state variance (~17% each), suggesting high collinearity
- [ ] Discuss with team: spending per establishment explains only ~4% of between-state variance and is not statistically significant; this may warrant revisiting the operationalization or examining whether spending is driving violations at all in this study period
- [ ] Continue building toward the full Aim 3 model by adding significant predictors jointly

---

*Script: `stepwise_zimmerman_models.py` | Level-2 covariates: `level2_covariates.csv`*
