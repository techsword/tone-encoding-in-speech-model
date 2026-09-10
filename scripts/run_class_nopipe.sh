#!/bin/bash

#SBATCH --output=run_class.%j.out
#SBATCH --job-name="class"


source .venv/bin/activate

ARG1=${1:-"facebook/wav2vec2-base"}

srun python -m tone_encoding.classification