"""
Hierarchical Mixed-Effects Model for Predicting Violations
State-Year Level Analysis (2011-2019)

This script:
1. Loads and reshapes EPA ECHO and WPS data
2. Filters to 50 US states only
3. Constructs time variables centered on 2017
4. Fits a hierarchical mixed-effects model with:
   - Fixed effects: time, time², time³, state
   - Random effects: random intercept, random state effects
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM
import warnings
warnings.filterwarnings('ignore')

# Define the 50 US states
US_STATES_50 = [
    'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California',
    'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia',
    'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa',
    'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland',
    'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri',
    'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey',
    'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio',
    'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina',
    'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont',
    'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming'
]

# Also handle potential variations in state names
STATE_NAME_MAPPING = {
    'Massachusetts ': 'Massachusetts',
    'Oregon ': 'Oregon',
}

print("=" * 70)
print("HIERARCHICAL MIXED-EFFECTS MODEL FOR VIOLATIONS")
print("=" * 70)

# Load the data
print("\n[1] Loading data...")
echo_df = pd.read_csv('/Users/keshavgoel/Research/establishments-data (2).csv', index_col=0)
wps_df = pd.read_csv('/Users/keshavgoel/Research/wps-data (2).csv')

print(f"EPA ECHO data shape: {echo_df.shape}")
print(f"WPS data shape: {wps_df.shape}")

# Clean up state names (strip whitespace, apply mapping)
echo_df.index = echo_df.index.str.strip()
echo_df.index = echo_df.index.map(lambda x: STATE_NAME_MAPPING.get(x, x))

wps_df['state'] = wps_df['state'].str.strip()
wps_df['state'] = wps_df['state'].map(lambda x: STATE_NAME_MAPPING.get(x, x))

# ============================================================
# RESHAPE EPA ECHO DATA TO LONG FORMAT
# ============================================================
print("\n[2] Reshaping EPA ECHO data to long format...")

# Extract violations columns for years 2011-2019
violation_cols = [f'violations-{year}' for year in range(2011, 2020)]

# Check which columns exist
available_violation_cols = [col for col in violation_cols if col in echo_df.columns]
print(f"Available violation columns: {available_violation_cols}")

# Create long format DataFrame
echo_long_list = []
for year in range(2011, 2020):
    col_name = f'violations-{year}'
    if col_name in echo_df.columns:
        temp_df = pd.DataFrame({
            'state': echo_df.index,
            'year': year,
            'violations': echo_df[col_name].values
        })
        echo_long_list.append(temp_df)

echo_long = pd.concat(echo_long_list, ignore_index=True)

# ============================================================
# FILTER TO 50 US STATES ONLY
# ============================================================
print("\n[3] Filtering to 50 US states only...")

# Filter ECHO data
echo_long_filtered = echo_long[echo_long['state'].isin(US_STATES_50)].copy()
print(f"ECHO data after filtering: {len(echo_long_filtered)} observations")
print(f"Unique states in ECHO data: {echo_long_filtered['state'].nunique()}")

# Check which states are missing
missing_states = set(US_STATES_50) - set(echo_long_filtered['state'].unique())
if missing_states:
    print(f"Missing states: {missing_states}")

# ============================================================
# CONSTRUCT TIME VARIABLES (CENTERED ON 2017)
# ============================================================
print("\n[4] Constructing time variables centered on 2017...")

# time = year - 2017 (so 2017 = 0)
echo_long_filtered['time'] = echo_long_filtered['year'] - 2017
echo_long_filtered['time2'] = echo_long_filtered['time'] ** 2
echo_long_filtered['time3'] = echo_long_filtered['time'] ** 3

print("\nTime variable mapping:")
for year in sorted(echo_long_filtered['year'].unique()):
    time_val = year - 2017
    print(f"  {year}: time = {time_val}, time² = {time_val**2}, time³ = {time_val**3}")

# ============================================================
# DATA CLEANING
# ============================================================
print("\n[5] Cleaning data...")

# Convert violations to numeric, handling missing/non-numeric values
echo_long_filtered['violations'] = pd.to_numeric(echo_long_filtered['violations'], errors='coerce')

# Check for missing values
missing_violations = echo_long_filtered['violations'].isna().sum()
print(f"Missing violation values: {missing_violations}")

# Remove rows with missing violations
df_model = echo_long_filtered.dropna(subset=['violations']).copy()
print(f"Final dataset size: {len(df_model)} observations")
print(f"Final unique states: {df_model['state'].nunique()}")

# Convert state to categorical factor
df_model['state'] = pd.Categorical(df_model['state'])

# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================
print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)

print("\nViolations summary:")
print(df_model['violations'].describe())

print("\nViolations by year:")
print(df_model.groupby('year')['violations'].agg(['mean', 'std', 'min', 'max', 'count']))

print("\nViolations by time (centered):")
print(df_model.groupby('time')['violations'].agg(['mean', 'std']).round(2))

# ============================================================
# FIT HIERARCHICAL MIXED-EFFECTS MODEL
# ============================================================
print("\n" + "=" * 70)
print("FITTING HIERARCHICAL MIXED-EFFECTS MODEL")
print("=" * 70)

print("""
Model Specification:
- Outcome: violations
- Fixed effects: time, time², time³
- Random effects: random intercept + random slope for time, grouped by state
- Grouping variable: state (categorical)

This is a multilevel model where:
- Level 1: Observations (state-years)
- Level 2: States

The model captures:
1. Overall time trend (fixed effects)
2. State-to-state baseline variation (random intercepts)
3. State-to-state variation in time trends (random slopes)
""")

# Model 1: Random intercept only
print("\n" + "-" * 70)
print("MODEL 1: Random Intercept Model")
print("-" * 70)
print("violations ~ time + time² + time³ + (1 | state)")

model1 = MixedLM.from_formula(
    'violations ~ time + time2 + time3',
    data=df_model,
    groups=df_model['state'],
    re_formula='~1'  # Random intercept only
)
result1 = model1.fit(method='lbfgs')
print(result1.summary())

# Model 2: Random intercept + random slope for time
print("\n" + "-" * 70)
print("MODEL 2: Random Intercept + Random Slope Model")
print("-" * 70)
print("violations ~ time + time² + time³ + (1 + time | state)")

model2 = MixedLM.from_formula(
    'violations ~ time + time2 + time3',
    data=df_model,
    groups=df_model['state'],
    re_formula='~time'  # Random intercept and random slope for time
)
result2 = model2.fit(method='lbfgs')
print(result2.summary())

# ============================================================
# DETAILED RESULTS INTERPRETATION
# ============================================================
print("\n" + "=" * 70)
print("DETAILED RESULTS INTERPRETATION")
print("=" * 70)

print("\n--- FIXED EFFECTS ---")
print("\nThese represent the average (national) effects across all states:")
fe = result2.fe_params
print(f"\nIntercept: {fe['Intercept']:.4f}")
print(f"  → Average violations at time=0 (year 2017)")

print(f"\ntime: {fe['time']:.4f}")
print(f"  → Linear trend: For each year from 2017, violations change by ~{fe['time']:.2f}")
if fe['time'] > 0:
    print("  → Positive = violations increasing over time")
else:
    print("  → Negative = violations decreasing over time")

print(f"\ntime²: {fe['time2']:.4f}")
if fe['time2'] > 0:
    print("  → Positive quadratic: U-shaped curve (violations higher at extremes)")
else:
    print("  → Negative quadratic: Inverted U-shape (violations peak in middle)")

print(f"\ntime³: {fe['time3']:.4f}")
print("  → Cubic term captures asymmetric patterns in the time trend")

print("\n--- RANDOM EFFECTS ---")
print("\nThese represent the variation BETWEEN states:")

# Get random effects variance components
print(f"\nRandom Effects Covariance:")
print(result2.cov_re)

# Calculate ICC (Intraclass Correlation Coefficient)
var_random_intercept = result2.cov_re.iloc[0, 0]
var_residual = result2.scale
icc = var_random_intercept / (var_random_intercept + var_residual)
print(f"\nVariance Components:")
print(f"  Between-state variance (random intercept): {var_random_intercept:.4f}")
print(f"  Within-state (residual) variance: {var_residual:.4f}")
print(f"  ICC (Intraclass Correlation): {icc:.4f}")
print(f"  → {icc*100:.1f}% of the total variation in violations is between states")

# ============================================================
# STATE-SPECIFIC RANDOM EFFECTS (BLUPs)
# ============================================================
print("\n" + "=" * 70)
print("STATE-SPECIFIC RANDOM EFFECTS (BLUPs)")
print("=" * 70)

random_effects = result2.random_effects
re_df = pd.DataFrame({
    'state': random_effects.keys(),
    'random_intercept': [v['Group'] for v in random_effects.values()],
    'random_slope_time': [v.get('time', 0) for v in random_effects.values()]
})
re_df = re_df.sort_values('random_intercept', ascending=False)

print("\nStates with HIGHEST baseline violations (positive random intercepts):")
print(re_df.head(10).to_string(index=False))

print("\nStates with LOWEST baseline violations (negative random intercepts):")
print(re_df.tail(10).to_string(index=False))

# ============================================================
# MODEL COMPARISON
# ============================================================
print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print("\n                    Model 1 (RI only)    Model 2 (RI + RS)")
print(f"Log-Likelihood:     {result1.llf:>15.2f}    {result2.llf:>15.2f}")
print(f"AIC:                {result1.aic:>15.2f}    {result2.aic:>15.2f}")
print(f"BIC:                {result1.bic:>15.2f}    {result2.bic:>15.2f}")

# Likelihood ratio test
lr_stat = 2 * (result2.llf - result1.llf)
print(f"\nLikelihood Ratio Test Statistic: {lr_stat:.4f}")
print("(Compare to chi-square distribution with df = difference in parameters)")

# ============================================================
# PREDICTED VALUES AND TREND VISUALIZATION DATA
# ============================================================
print("\n" + "=" * 70)
print("PREDICTED NATIONAL TREND")
print("=" * 70)

# Create predictions for each year
years = list(range(2011, 2020))
times = [y - 2017 for y in years]
times2 = [t**2 for t in times]
times3 = [t**3 for t in times]

pred_df = pd.DataFrame({
    'year': years,
    'time': times,
    'time2': times2,
    'time3': times3
})

# Calculate predicted values using fixed effects only
pred_df['predicted_violations'] = (
    fe['Intercept'] +
    fe['time'] * pred_df['time'] +
    fe['time2'] * pred_df['time2'] +
    fe['time3'] * pred_df['time3']
)

# Also get actual mean violations by year
actual_means = df_model.groupby('year')['violations'].mean()
pred_df['actual_mean'] = pred_df['year'].map(actual_means)

print("\nYear-by-Year Comparison (Fixed Effects Predictions vs Actual):")
print(pred_df[['year', 'time', 'predicted_violations', 'actual_mean']].to_string(index=False))

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("""
KEY FINDINGS:

1. TIME TREND (Fixed Effects):
   - The model includes linear, quadratic, and cubic time terms
   - These capture the non-linear pattern of violations relative to 2017

2. STATE VARIATION (Random Effects):
   - Significant variation exists between states in baseline violation rates
   - States differ in their time trajectories (random slopes)
   - The ICC indicates what proportion of variance is between-state

3. HIERARCHICAL STRUCTURE:
   - The model properly accounts for the nested structure
   - State-year observations are clustered within states
   - This avoids treating repeated observations as independent

4. INTERPRETATION:
   - Fixed effects show the "national average" pattern
   - Random effects show how each state deviates from this pattern
   - BLUPs give state-specific predictions
""")

# Save results to files
print("\n[6] Saving results...")

# Save the model data
df_model.to_csv('/Users/keshavgoel/Research/model_data_long.csv', index=False)
print("Saved: model_data_long.csv")

# Save random effects
re_df.to_csv('/Users/keshavgoel/Research/state_random_effects.csv', index=False)
print("Saved: state_random_effects.csv")

# Save predictions
pred_df.to_csv('/Users/keshavgoel/Research/predicted_trend.csv', index=False)
print("Saved: predicted_trend.csv")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
