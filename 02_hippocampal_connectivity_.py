#!/usr/bin/env python
"""Clean FC matrices and compute participation coefficient."""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import bct

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

input_dir = Path('outputs/FC_full_batch_outputs')
output_dir = Path('outputs/FC_full_outputs_clean')
output_dir.mkdir(parents=True, exist_ok=True)

exclude_regions = [str(r) for r in config['fc_cleaning']['exclude_regions']]
fc_files = sorted(input_dir.glob('*.csv'))

print(f"Cleaning {len(fc_files)} FC matrices...")

# Clean FC matrices
for fc_path in fc_files:
    fc_df = pd.read_csv(fc_path, index_col=0)
    fc_df.index = fc_df.index.astype(str)
    fc_df.columns = fc_df.columns.astype(str)
    
    fc_df_clean = fc_df.drop(
        index=exclude_regions,
        columns=exclude_regions,
        errors='ignore'
    )
    
    out_path = output_dir / fc_path.name
    fc_df_clean.to_csv(out_path)

print("Done cleaning.")

# Compute participation coefficient
print("\nComputing participation coefficient...")

yeo_file = 'data/SchaeferMAGeT_to_Yeo.xlsx'
yeo_df = pd.read_excel(yeo_file, header=None)
ci = yeo_df.iloc[1, :].values.astype(int)

fc_files = sorted(output_dir.glob('*.csv'))
partcoef_data = []
eids = []

for fc_path in fc_files:
    eid = fc_path.name.replace('sub-', '').replace('_full_correlation_matrix.csv', '')
    fc_df = pd.read_csv(fc_path, index_col=0)
    fc_abs = fc_df.abs()
    
    pc = bct.participation_coef(fc_abs.values, ci)
    partcoef_data.append(pc)
    eids.append(eid)

pc_df = pd.DataFrame(partcoef_data, index=eids)
pc_df.index.name = 'eid'

pc_df.to_csv('outputs/ParticipationCoefficient.csv')
pc_df.to_excel('outputs/ParticipationCoefficient.xlsx')

print(f"Saved participation coefficients")
