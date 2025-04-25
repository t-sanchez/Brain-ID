#!/bin/bash

#SBATCH --job-name=train_brain-id
#SBATCH --gpus=1
#SBATCH --partition=  

#SBATCH --mail-type=FAIL
#SBATCH --account= 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G # 256G
#SBATCH --time=6-23:59:59
#SBATCH --output=logs/%j.log # Standard output and error log 


# exp-specific cfg #
exp_cfg_file=train_feature_fetal

# Get the current time from command line
# Get the current date and time

echo python scripts/train_feature_fetal.py --config-name $exp_cfg_file
python scripts/train_feature_fetal.py --config-name $exp_cfg_file 
date
