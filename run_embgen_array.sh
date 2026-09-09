#!/bin/bash

#SBATCH --output=generate-%A_%a.out
#SBATCH --job-name="generate"
#SBATCH --gres=gpu:1
#SBATCH --array=0-8%2


source /usr/local/anaconda3/etc/profile.d/conda.sh

conda activate tones

    # "kehanlu/mandarin-wav2vec2" 
    # "kehanlu/mandarin-wav2vec2-aishell1"
    # "facebook/wav2vec2-base"
    # "facebook/wav2vec2-base-960h"
    # "facebook/wav2vec2-large"
    # "facebook/hubert-base-ls960" 
    # "facebook/hubert-large-ll60k"
    # "facebook/wav2vec2-large-xlsr-53"
    # "TencentGameMate/chinese-hubert-base"
    # "TencentGameMate/chinese-hubert-large"
    # "TencentGameMate/chinese-wav2vec2-base"
    # "TencentGameMate/chinese-wav2vec2-large"
    # "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
    # "nguyenvulebinh/wav2vec2-base-vi"
    # "nguyenvulebinh/wav2vec2-base-vi-vlsp2020"
    # "patrickvonplaten/wav2vec2-base-random"
    # "LeBenchmark/wav2vec2-FR-7K-base"

model_names=(
    "facebook/wav2vec2-base"
    "facebook/wav2vec2-base-960h"
    "kehanlu/mandarin-wav2vec2" 
    "kehanlu/mandarin-wav2vec2-aishell1"
    "nguyenvulebinh/wav2vec2-base-vi"
    "nguyenvulebinh/wav2vec2-base-vi-vlsp2020"
    "wcfr/wav2vec2-conformer-rel-pos-base-cantonese"
    "LeBenchmark/wav2vec2-FR-7K-base"
    "TencentGameMate/chinese-wav2vec2-base"
    )




srun python generate_classifier_input.py --model_name ${model_names[$SLURM_ARRAY_TASK_ID]} --dataset_name thchs30
srun python generate_classifier_input.py --model_name ${model_names[$SLURM_ARRAY_TASK_ID]} --dataset_name vivos