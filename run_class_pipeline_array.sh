#!/bin/bash

#SBATCH --output=run_pipeline-%A_%a.out
#SBATCH --job-name="run_pipeline"
#SBATCH --gres=gpu:1
#SBATCH --array=0-12%2


source /usr/local/anaconda3/etc/profile.d/conda.sh

conda activate tones

model_names=(
	"facebook/wav2vec2-base"
	"bert-base-chinese"
	"kehanlu/mandarin-wav2vec2"
	"TencentGameMate/chinese-wav2vec2-base"
	"facebook/wav2vec2-base-960h"
	"kehanlu/mandarin-wav2vec2-aishell1"
	"LeBenchmark/wav2vec2-FR-1K-base"
	"LeBenchmark/wav2vec2-FR-2.6K-base"
	"LeBenchmark/wav2vec2-FR-3K-base"
	"LeBenchmark/wav2vec2-FR-7K-base"
	"nguyenvulebinh/wav2vec2-base-vi"
    "nguyenvulebinh/wav2vec2-base-vi-vlsp2020"
	"wcfr/wav2vec2-conformer-rel-pos-base-cantonese"
    )


# Get the index from SLURM_ARRAY_TASK_ID
index=$SLURM_ARRAY_TASK_ID

# Get the name at the specified index
current_name=${model_names[$index]}

echo "Running basic experiment for ${current_name}"

# Running the basic experiment
srun python classification_pipeline.py --model_name $current_name --mode 'heldout' --contrast 'tone'
srun python classification_pipeline.py --model_name $current_name --mode 'heldout' --contrast 'consonant'
# Running the subclass experiment
srun python classification_pipeline.py --model_name $current_name --mode 'heldout' --contrast 'tone' --subclass
srun python classification_pipeline.py --model_name $current_name --mode 'heldout' --contrast 'consonant' --subclass

# Running the basic experiment on Vietnamese data 'vivos'
srun python classification_pipeline.py --model_name $current_name --dataset 'vivos' --mode 'heldout' --contrast 'tone'


echo "FINISHED!"