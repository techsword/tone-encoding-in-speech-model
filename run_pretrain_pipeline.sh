#!/bin/bash

#SBATCH --output=run_pipeline-%A_%a.out
#SBATCH --job-name="run_pipeline"
#SBATCH --array=0-17%4


source /usr/local/anaconda3/etc/profile.d/conda.sh

conda activate tones

# find fairseq-pretrained-models/*/*5000.pt | sort -V

model_names=(
fairseq-pretrained-models/wav2vec2_base_librispeech/checkpoint_15_5000.pt
fairseq-pretrained-models/wav2vec2_base_librispeech/checkpoint_45_15000.pt
fairseq-pretrained-models/wav2vec2_base_librispeech/checkpoint_75_25000.pt
fairseq-pretrained-models/wav2vec2_base_librispeech/checkpoint_105_35000.pt
fairseq-pretrained-models/wav2vec2_base_librispeech/checkpoint_134_45000.pt
fairseq-pretrained-models/wav2vec2_base_librispeech/checkpoint_164_55000.pt
fairseq-pretrained-models/wav2vec2_base_librispeech/checkpoint_194_65000.pt
fairseq-pretrained-models/wav2vec2_base_librispeech/checkpoint_224_75000.pt
fairseq-pretrained-models/wav2vec2_base_librispeech/checkpoint_254_85000.pt
fairseq-pretrained-models/wav2vec2_base_magicdata/checkpoint_16_5000.pt
fairseq-pretrained-models/wav2vec2_base_magicdata/checkpoint_48_15000.pt
fairseq-pretrained-models/wav2vec2_base_magicdata/checkpoint_79_25000.pt
fairseq-pretrained-models/wav2vec2_base_magicdata/checkpoint_111_35000.pt
fairseq-pretrained-models/wav2vec2_base_magicdata/checkpoint_142_45000.pt
fairseq-pretrained-models/wav2vec2_base_magicdata/checkpoint_174_55000.pt
fairseq-pretrained-models/wav2vec2_base_magicdata/checkpoint_206_65000.pt
fairseq-pretrained-models/wav2vec2_base_magicdata/checkpoint_237_75000.pt
fairseq-pretrained-models/wav2vec2_base_magicdata/checkpoint_269_85000.pt
    )

srun python generate_classifier_input.py --model_name ${model_names[$SLURM_ARRAY_TASK_ID]} --dataset_name thchs30

srun python classification_pipeline.py --model_name ${model_names[$SLURM_ARRAY_TASK_ID]} --mode 'heldout' --contrast 'tone' --results_path "results/pretrained_pipeline_results"
srun python classification_pipeline.py --model_name ${model_names[$SLURM_ARRAY_TASK_ID]} --mode 'heldout' --contrast 'consonant' --results_path "results/pretrained_pipeline_results"


srun python classification_pipeline.py --model_name ${model_names[$SLURM_ARRAY_TASK_ID]} --mode 'heldout' --contrast 'tone' --results_path "results/pretrained_pipeline_results" --subclass
srun python classification_pipeline.py --model_name ${model_names[$SLURM_ARRAY_TASK_ID]} --mode 'heldout' --contrast 'consonant' --results_path "results/pretrained_pipeline_results" --subclass

