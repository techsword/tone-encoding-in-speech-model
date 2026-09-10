#!/bin/bash

#SBATCH --output=run_pipeline-%A_%a.out
#SBATCH --job-name="run_pipeline"
#SBATCH --array=0-17%4


source .venv/bin/activate

# The pretrained checkpoints are loaded from the Hugging Face Hub as
# safetensors, not as local fairseq .pt files. Each training checkpoint is a
# branch (revision) of a hub repo:
#   techsword/wav2vec2-base-english-librispeech730h  branches ckpt-5000 .. ckpt-85000
#   techsword/wav2vec2-base-mandarin-magicdata       branches ckpt-5000 .. ckpt-85000
# Pass the repo as --model_name and the branch as --revision. The loader
# (generate_classifier_input.loading_pretrained_model) imports the model as a
# torchaudio Wav2Vec2Model. No download step is needed; the hub fetches the
# branch on first use.

model_names=(
techsword/wav2vec2-base-english-librispeech730h
techsword/wav2vec2-base-english-librispeech730h
techsword/wav2vec2-base-english-librispeech730h
techsword/wav2vec2-base-english-librispeech730h
techsword/wav2vec2-base-english-librispeech730h
techsword/wav2vec2-base-english-librispeech730h
techsword/wav2vec2-base-english-librispeech730h
techsword/wav2vec2-base-english-librispeech730h
techsword/wav2vec2-base-english-librispeech730h
techsword/wav2vec2-base-mandarin-magicdata
techsword/wav2vec2-base-mandarin-magicdata
techsword/wav2vec2-base-mandarin-magicdata
techsword/wav2vec2-base-mandarin-magicdata
techsword/wav2vec2-base-mandarin-magicdata
techsword/wav2vec2-base-mandarin-magicdata
techsword/wav2vec2-base-mandarin-magicdata
techsword/wav2vec2-base-mandarin-magicdata
techsword/wav2vec2-base-mandarin-magicdata
    )

revisions=(
ckpt-5000
ckpt-15000
ckpt-25000
ckpt-35000
ckpt-45000
ckpt-55000
ckpt-65000
ckpt-75000
ckpt-85000
ckpt-5000
ckpt-15000
ckpt-25000
ckpt-35000
ckpt-45000
ckpt-55000
ckpt-65000
ckpt-75000
ckpt-85000
    )

srun python -m tone_encoding.generate_classifier_input --model_name ${model_names[$SLURM_ARRAY_TASK_ID]} --revision ${revisions[$SLURM_ARRAY_TASK_ID]} --dataset_name thchs30

srun python -m tone_encoding.classification_pipeline --model_name ${model_names[$SLURM_ARRAY_TASK_ID]} --revision ${revisions[$SLURM_ARRAY_TASK_ID]} --mode 'heldout' --contrast 'tone' --results_path "results/pretrained_pipeline_results"
srun python -m tone_encoding.classification_pipeline --model_name ${model_names[$SLURM_ARRAY_TASK_ID]} --revision ${revisions[$SLURM_ARRAY_TASK_ID]} --mode 'heldout' --contrast 'consonant' --results_path "results/pretrained_pipeline_results"


srun python -m tone_encoding.classification_pipeline --model_name ${model_names[$SLURM_ARRAY_TASK_ID]} --revision ${revisions[$SLURM_ARRAY_TASK_ID]} --mode 'heldout' --contrast 'tone' --results_path "results/pretrained_pipeline_results" --subclass
srun python -m tone_encoding.classification_pipeline --model_name ${model_names[$SLURM_ARRAY_TASK_ID]} --revision ${revisions[$SLURM_ARRAY_TASK_ID]} --mode 'heldout' --contrast 'consonant' --results_path "results/pretrained_pipeline_results" --subclass

