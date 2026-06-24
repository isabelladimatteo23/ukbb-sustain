#!/usr/bin/env python
"""Extract functional connectivity from fMRIPrep data."""

import sys
import re
from pathlib import Path
import pandas as pd
import numpy as np
from nilearn import image as nimg
from nilearn import input_data
from nilearn.connectome import ConnectivityMeasure
from bids import BIDSLayout
from bids.layout import BIDSLayoutIndexer
import yaml

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

TASK_ID = int(sys.argv[1])
BATCH_SIZE = config['batch']['batch_size']
start = TASK_ID * BATCH_SIZE
end = start + BATCH_SIZE

# Load subjects
subjects = pd.read_csv('data/ukbb_imaging_subjects.csv', dtype=str)['eid'].tolist()
batch = subjects[start:end]
print(f"Task {TASK_ID}: Processing {len(batch)} subjects")

# Setup BIDS layout
fmriprep_dir = config['paths']['fmriprep_dir']
ignore_pattern = re.compile(r"sub-(?!" + "|".join(batch) + r")\d+")
indexer = BIDSLayoutIndexer(validate=False, ignore=[ignore_pattern], index_metadata=False)
layout = BIDSLayout(fmriprep_dir, indexer=indexer, validate=False, derivatives=False)
subjects = layout.get_subjects()

# Load atlas
parcel_file = 'data/atlas-Schaefer100_plus_MAGeT_dseg_v2.nii.gz'
parcel_img = nimg.load_img(parcel_file)

# Setup masker
masker = input_data.NiftiLabelsMasker(
    labels_img=parcel_img,
    standardize=True,
    memory='nilearn_cache',
    verbose=1,
    detrend=True,
    low_pass=config['fmri']['low_pass'],
    high_pass=config['fmri']['high_pass'],
    t_r=config['fmri']['tr']
)

# Setup connectivity measure
correlation_measure = ConnectivityMeasure(kind='correlation')

# Output directory
output_dir = Path('outputs/FC_full_batch_outputs')
output_dir.mkdir(parents=True, exist_ok=True)

def extract_confounds(confound_tsv, confounds, dt=True):
    all_confounds = confounds.copy()
    if dt:
        dt_names = [f'{c}_derivative1' for c in confounds]
        all_confounds += dt_names
    confound_df = pd.read_csv(confound_tsv, delimiter='\t')
    return confound_df[all_confounds].values

# Process subjects
tr_drop = config['fmri']['tr_drop']
confound_variables = config['fmri']['confound_variables']

for sub in subjects:
    func_files = [f for f in layout.get(subject=sub, datatype='func', task='rest',
                                       extension=['.nii', '.nii.gz'], return_type='file')
                  if 'preproc_bold' in f]
    
    confound_files = [f for f in layout.get(subject=sub, datatype='func', task='rest',
                                            extension='.tsv', return_type='file')
                      if 'confounds' in f]
    
    if not func_files or not confound_files:
        continue
    
    func_img = nimg.load_img(func_files[0]).slicer[:, :, :, tr_drop:]
    confounds = extract_confounds(confound_files[0], confound_variables)[tr_drop:, :]
    time_series = masker.fit_transform(func_img, confounds)
    full_correlation_matrix = correlation_measure.fit_transform([time_series])[0]
    
    roi_labels = masker.labels_
    df = pd.DataFrame(full_correlation_matrix, index=roi_labels, columns=roi_labels)
    
    out_path = output_dir / f"sub-{sub}_full_correlation_matrix.csv"
    if not out_path.exists():
        df.to_csv(out_path)
        print(f"Saved: {out_path}")
