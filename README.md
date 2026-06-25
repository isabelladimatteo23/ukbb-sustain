# UKBB Functional Connectivity & SuStaIn Pipeline

A pipeline for computing functional connectivity, hippocampal participation coefficients, and sustain disease progression modeling.

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Place data files in `data/` directory
3. Update paths in config.yaml
4. Run scripts in order

## Scripts

- `01_fc_batch.py` - Extract functional connectivity
- `02_hippocampal_connectivity.py` - Clean FC and compute participation coefficient  
- `03_sustain_cv.py` - Run SuStaIn modeling

## Data Requirements

Place in `data/` directory:
- `ukbb_imaging_subjects.csv`
- `ukbb_imaging_filtered.csv`
- `ukbb_imaging_with_HC_PC_and_eTIV.csv`
- Atlas files: `atlas-Schaefer100_plus_MAGeT_dseg_v2.*`
- `SchaeferMAGeT_to_Yeo.xlsx`
