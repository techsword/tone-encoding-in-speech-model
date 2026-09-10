#!/bin/bash

#SBATCH --output=embgen.%j.out
#SBATCH --job-name="embgen"
#SBATCH --gres=gpu:1


source .venv/bin/activate

ARG1=${1:-"facebook/wav2vec2-base"}

ARG2=${2:-"thchs30"}
# srun python -m tone_encoding.generate_classifier_input --model_name $ARG1 --flattened 
# srun python -m tone_encoding.generate_classifier_input --model_name $ARG1 --cnn 
# srun python -m tone_encoding.generate_classifier_input --model_name $ARG1 --flattened --cnn 
#srun python -m tone_encoding.generate_classifier_input --model_name $ARG1
srun python -m tone_encoding.generate_classifier_input --model_name $ARG1 --dataset_name $ARG2

