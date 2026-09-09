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
tone_encoding.generate_aligned_dataset  ->  tone_encoding.generate_classifier_input  ->  tone_encoding.classification_pipeline  ->  tone_encoding.experiment_classification
```

1. `tone_encoding.generate_aligned_dataset` — read Charsiu-produced TextGrid
   alignments and the corpus transcriptions, and write the aligned CSV datasets
   used by the rest of the pipeline.
2. `tone_encoding.generate_classifier_input` — run a pretrained model over the
   audio and save the per-word segment embeddings plus labels (the classifier
   input).
3. `tone_encoding.classification_pipeline` — CLI wrapper that chains embedding
   generation and classification.
4. `tone_encoding.experiment_classification` — run the ridge classifier (tone /
   consonant contrast, heldout or all-data split, optional subclass
   experiments).

Analysis and plotting:

- `tone_encoding.plot_results` — plots for the paper.
- `tone_encoding.visualizations` — additional plotting helpers.

Audio-feature baselines (f0, MFCC) are computed in
`tone_encoding.audiofeat_preprocess`, `tone_encoding.generate_audiofeature`, and
`tone_encoding.audiofeat_pitch_manipulation`.

## Repository layout

```
src/tone_encoding/   Python package: pipeline and analysis modules
scripts/             SLURM scripts that orchestrate the pipeline stages
data/epoch_maps/     Epoch-to-update mapping tables used by tone_encoding.plot_results
```

| Module | Purpose |
|---|---|
| `tone_encoding.generate_aligned_dataset` | Read TextGrids + transcriptions, write aligned CSVs |
| `tone_encoding.generate_classifier_input` | Extract segment embeddings from a pretrained model |
| `tone_encoding.classification_pipeline` | Chain embedding generation + classification (CLI) |
| `tone_encoding.experiment_classification` | Ridge classification experiments |
| `tone_encoding.classification` | Legacy predecessor of `experiment_classification` |
| `tone_encoding.preprocessing` | THCHS-30 / VIVOS TextGrid preprocessing helpers |
| `tone_encoding.audiofeat_preprocess`, `tone_encoding.generate_audiofeature`, `tone_encoding.audiofeat_pitch_manipulation` | f0 / MFCC baselines and pitch flattening |
| `tone_encoding.dataset_insights` | Dataset statistics helpers |
| `tone_encoding.plot_results`, `tone_encoding.visualizations` | Result plotting |
| `tone_encoding.baselines_sanity_check` | MLP baseline sanity check |

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
`scripts/run_pretrain_pipeline.sh`); it is treated as a local fairseq file when
the path exists on disk and contains "fairseq"
(see `tone_encoding.generate_classifier_input.load_fairseq_model`).

## Environment setup

`uv` manages the environment. Most dependencies are pinned in
`pyproject.toml`. Python 3.10 is required: fairseq 0.12.2 does not import on
Python 3.11 or later.

PyTorch is provided through two mutually exclusive extras. Select exactly one:

```bash
# CPU only (local development)
uv sync --extra cpu

# NVIDIA GPU with CUDA 12.1 (HPC)
uv sync --extra cu121
```

`uv sync` creates `.venv/` and installs the pinned dependency set, including
fairseq 0.12.2. It also installs the `tone_encoding` package from `src/` in
editable mode, so both `python -m tone_encoding.X` and
`from tone_encoding import X` work with no `sys.path` changes. The `cu121` extra
matches the validated 2024 environment. Do not select both extras.

No `uv.lock` is committed. Torch resolution depends on the chosen accelerator
extra, so a committed lock would pin one backend for everyone.

To run a module directly, use `uv run` from the repository root:

```bash
uv run python -m tone_encoding.generate_classifier_input --model_name facebook/wav2vec2-base --dataset_name thchs30
```

The SLURM scripts in `scripts/` activate the environment with
`source .venv/bin/activate`. Submit them from the repository root.

## Usage

The SLURM scripts in `scripts/` run the pipeline stages:

- `scripts/run_embgen.sh` / `scripts/run_embgen_array.sh` — generate classifier
  input with `python -m tone_encoding.generate_classifier_input`.
- `scripts/run_class_pipeline_array.sh` — run classification with
  `python -m tone_encoding.classification_pipeline` across a model list.
- `scripts/run_pretrain_pipeline.sh` — run the pretrained-checkpoint
  experiments.
- `scripts/run_class_nopipe.sh` — run the legacy
  `python -m tone_encoding.classification`.

The main CLI entry points are (run from the repository root):

```bash
python -m tone_encoding.generate_classifier_input --model_name facebook/wav2vec2-base --dataset_name thchs30
python -m tone_encoding.classification_pipeline --model_name facebook/wav2vec2-base --mode heldout --contrast tone
```

`tone_encoding.classification_pipeline` accepts `--mode {heldout,alldata}` and
`--contrast {tone,consonant}`, plus optional `--subclass`, `--segment_input`,
`--flattened`, `--cnn`, `--tgt_layers`, and `--seed` flags.

## Legacy scripts

`tone_encoding.classification` is kept as a legacy predecessor of
`tone_encoding.experiment_classification`;
`tone_encoding.baselines_sanity_check` and `tone_encoding.plot_results` import
helper functions from it.

## License

Apache-2.0. See [LICENSE](LICENSE).
