#!/bin/bash
#SBATCH --job-name=fc_extraction
#SBATCH --array=0-86
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH --output=logs/batch_%A_%a.out

module load scipy-stack/2025a
export PYTHONPATH=$HOME/python_pkgs/lib/python3.11/site-packages:$PYTHONPATH

python 01_fc_batch.py $SLURM_ARRAY_TASK_ID
