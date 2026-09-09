#!/bin/bash

#SBATCH --output=embgen.%j.out
#SBATCH --job-name="embgen"
#SBATCH --gres=gpu:1


source /usr/local/anaconda3/etc/profile.d/conda.sh

conda activate tones

ARG1=${1:-"facebook/wav2vec2-base"}

ARG2=${2:-"thchs30"}
# srun python generate_classifier_input.py --model_name $ARG1 --flattened 
# srun python generate_classifier_input.py --model_name $ARG1 --cnn 
# srun python generate_classifier_input.py --model_name $ARG1 --flattened --cnn 
#srun python generate_classifier_input.py --model_name $ARG1
srun python generate_classifier_input.py --model_name $ARG1 --dataset_name $ARG2

