#!/usr/bin/env python
"""Run SuStaIn disease progression modeling with cross-validation."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pySuStaIn
import statsmodels.formula.api as smf
from sklearn.model_selection import StratifiedKFold
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

np.random.seed(config['random_state'])

# Load and prepare data
df = pd.read_csv('data/ukbb_imaging_with_HC_PC_and_eTIV.csv')
biomarkers = config['biomarkers']
covariates = config['covariates']

analysis_cols = biomarkers + covariates + ['dx']
df = df.dropna(subset=analysis_cols).copy()
df = df.set_index('eid')

print(f"Loaded {len(df)} subjects")

# Z-score normalize biomarkers against controls
for b in biomarkers:
    mod = smf.ols(f"{b} ~ age + sex + eTIV", data=df[df['dx'] == 0]).fit()
    predicted = mod.predict(df[['age', 'sex', 'eTIV']])
    df[b] = (df[b] - predicted) / mod.resid.std()

# Remove outliers
is_outlier = df[biomarkers].abs().gt(10).any(axis=1)
df = df.drop(df[is_outlier].index)
print(f"Final sample: {len(df)} ({(df['dx']==0).sum()} controls, {(df['dx']==1).sum()} patients)")

# Invert biomarkers (higher = worse)
for b in biomarkers:
    if b in ['left_HC_participation', 'right_HC_participation', 'pal', 'freq_social_visits']:
        df[b] *= -1

# Prepare SuStaIn inputs
N = len(biomarkers)
Z_vals = np.array([[1, 2, 3]] * N)
Z_max = np.array([5] * N)

output_folder = 'outputs/sustain_output'

sustain_input = pySuStaIn.ZscoreSustain(
    df[biomarkers].values,
    Z_vals,
    Z_max,
    biomarkers,
    config['sustain']['n_startpoints'],
    config['sustain']['n_s_max'],
    int(config['sustain']['n_iterations_mcmc']),
    output_folder,
    'sustain_output',
    True,
)

# Run SuStaIn
print("\nRunning SuStaIn...")
samples_sequence, samples_f, ml_subtype, prob_ml_subtype, ml_stage, prob_ml_stage, prob_subtype_stage = sustain_input.run_sustain_algorithm()

# Cross-validation
print("\nPerforming cross-validation...")
n_folds = config['sustain']['cv_n_folds']
labels = df['dx'].values

cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config['sustain']['cv_random_state'])
test_idxs = [test for train, test in cv.split(df, labels)]
test_idxs = np.array(test_idxs, dtype='object')

CVIC, loglike_matrix = sustain_input.cross_validate_sustain_model(test_idxs)

print(f"CVIC: {CVIC}")
print(f"Mean test log-likelihood: {np.mean(loglike_matrix, 0)}")

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(np.arange(config['sustain']['n_s_max']), CVIC, marker='o')
plt.xlabel('Number of subtypes')
plt.ylabel('CVIC')
plt.title('Model selection')
plt.savefig('outputs/CVIC.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nDone! Check outputs/ directory for results.")
