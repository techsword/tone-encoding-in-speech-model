#!/bin/bash

#SBATCH --output=run_class.%j.out
#SBATCH --job-name="class"


source /usr/local/anaconda3/etc/profile.d/conda.sh

conda activate tones

ARG1=${1:-"facebook/wav2vec2-base"}

srun python classification.py