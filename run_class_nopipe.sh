#!/bin/bash

#SBATCH --output=run_class.%j.out
#SBATCH --job-name="class"


source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate tones

ARG1=${1:-"facebook/wav2vec2-base"}

srun python classification.py