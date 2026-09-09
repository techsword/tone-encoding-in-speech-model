# Encoding of lexical tone in self-supervised models of spoken language

Code for the NAACL 2024 paper:

> **Encoding of lexical tone in self-supervised models of spoken language**
> Gaofei Shen, Michaela Watkins, Afra Alishahi, Arianna Bisazza, Grzegorz Chrupała.
> NAACL 2024. arXiv: [2403.16865](https://arxiv.org/abs/2403.16865).

This repository contains the experiment code used to produce the paper's
results. It probes how self-supervised spoken-language models encode lexical
tone in Mandarin (THCHS-30) and Vietnamese (VIVOS).

## Pipeline

The main processing pipeline is:

```
generate_aligned_dataset.py  ->  generate_classifier_input.py  ->  classification_pipeline.py  ->  experiment_classification.py
```

1. `generate_aligned_dataset.py` — read Charsiu-produced TextGrid alignments
   and the corpus transcriptions, and write the aligned CSV datasets used by
   the rest of the pipeline.
2. `generate_classifier_input.py` — run a pretrained model over the audio and
   save the per-word segment embeddings plus labels (the classifier input).
3. `classification_pipeline.py` — CLI wrapper that chains embedding generation
   and classification.
4. `experiment_classification.py` — run the ridge classifier (tone / consonant
   contrast, heldout or all-data split, optional subclass experiments).

Analysis and plotting:

- `plot_results.py` — plots for the paper.
- `visualizations.py` — additional plotting helpers.

Audio-feature baselines (f0, MFCC) are computed in `audiofeat_preprocess.py`,
`generate_audiofeature.py`, and `audiofeat_pitch_manipulation.py`.

## Repository layout

| File | Purpose |
|---|---|
| `generate_aligned_dataset.py` | Read TextGrids + transcriptions, write aligned CSVs |
| `generate_classifier_input.py` | Extract segment embeddings from a pretrained model |
| `classification_pipeline.py` | Chain embedding generation + classification (CLI) |
| `experiment_classification.py` | Ridge classification experiments |
| `classification.py` | Legacy predecessor of `experiment_classification.py` |
| `preprocessing.py` | THCHS-30 / VIVOS TextGrid preprocessing helpers |
| `audiofeat_preprocess.py`, `generate_audiofeature.py`, `audiofeat_pitch_manipulation.py` | f0 / MFCC baselines and pitch flattening |
| `dataset_insights.py` | Dataset statistics helpers |
| `plot_results.py`, `visualizations.py` | Result plotting |
| `baselines_sanity_check.py` | MLP baseline sanity check |
| `run_*.sh` | SLURM scripts that orchestrate the pipeline stages |

## Data requirements

The corpora are **not** bundled in this repo. Alignment is external: the
experiments require Charsiu-produced TextGrid files. You need:

- **THCHS-30** (Mandarin) — audio, `.wav.trn` transcriptions, and TextGrid
  alignments.
- **VIVOS** (Vietnamese) — audio, `prompts_ipa.txt`, and TextGrid alignments.
- **LibriSpeech** (English, for cross-lingual checks).
- **MAGICDATA** (Mandarin, for the pretrained-checkpoint experiments).

Corpus locations are configurable via environment variables
(`CORPORA_ROOT`, `THCHS30_DIR`, `VIVOS_DIR`). Defaults are repo-relative under
`corpora/`. For example:

```bash
export CORPORA_ROOT=/path/to/corpora
export THCHS30_DIR=/path/to/data_thchs30
export VIVOS_DIR=/path/to/vivos
```

## Pretrained fairseq checkpoints

The raw fairseq checkpoints used for the pretrained-checkpoint experiments are
not stored in this repo. Download them from Hugging Face into
`fairseq-pretrained-models/`, keeping the original subdirectory layout:

- Raw checkpoints:
  - `techsword/wav2vec2-base-english-librispeech730h-checkpoints`
  - `techsword/wav2vec2-base-mandarin-magicdata-checkpoints`
- Converted HF-format models (for direct use with the transformers library):
  - `techsword/wav2vec2-base-english-librispeech730h`
  - `techsword/wav2vec2-base-mandarin-magicdata`

The checkpoint path is passed to the loader (e.g. `--model_name` in
`run_pretrain_pipeline.sh`); it is treated as a local fairseq file when the
path exists on disk and contains "fairseq"
(see `generate_classifier_input.load_fairseq_model`).

## Environment setup

`uv` manages the environment. Dependencies are pinned in `pyproject.toml` and
`uv.lock`. Python 3.10 is required: fairseq 0.12.2 does not import on Python
3.11 or later.

```bash
uv sync
```

`uv sync` creates `.venv/` and installs the pinned dependency set, including
fairseq 0.12.2. The default wheels are CUDA-enabled, matching the original
2024 environment. To run a script directly, use `uv run`:

```bash
uv run python generate_classifier_input.py --model_name facebook/wav2vec2-base --dataset_name thchs30
```

The SLURM scripts activate the environment with `source .venv/bin/activate`.
Submit them from the repository root.

## Usage

The SLURM scripts run the pipeline stages:

- `run_embgen.sh` / `run_embgen_array.sh` — generate classifier input with
  `generate_classifier_input.py`.
- `run_class_pipeline_array.sh` — run classification with
  `classification_pipeline.py` across a model list.
- `run_pretrain_pipeline.sh` — run the pretrained-checkpoint experiments.
- `run_class_nopipe.sh` — run the legacy `classification.py`.

The main CLI entry points are:

```bash
python generate_classifier_input.py --model_name facebook/wav2vec2-base --dataset_name thchs30
python classification_pipeline.py --model_name facebook/wav2vec2-base --mode heldout --contrast tone
```

`classification_pipeline.py` accepts `--mode {heldout,alldata}` and
`--contrast {tone,consonant}`, plus optional `--subclass`, `--segment_input`,
`--flattened`, `--cnn`, `--tgt_layers`, and `--seed` flags.

## Legacy scripts

`classification.py` is kept as a legacy predecessor of
`experiment_classification.py`; `baselines_sanity_check.py` and
`plot_results.py` import helper functions from it.

## License

Apache-2.0. See [LICENSE](LICENSE).
