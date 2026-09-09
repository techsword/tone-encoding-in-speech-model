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

## Pretrained checkpoints (Hugging Face safetensors)

The pretrained-checkpoint experiments load their models from the Hugging Face
Hub as safetensors. Each training checkpoint is a branch (revision) of a repo:

- `techsword/wav2vec2-base-english-librispeech730h`
- `techsword/wav2vec2-base-mandarin-magicdata`

Branches are named `ckpt-<step>`: `ckpt-5000`, `ckpt-15000`, ..., `ckpt-85000`
(the odd 5000-step checkpoints). The `main` branch holds the best checkpoint.

Pass the repo as `--model_name` and the branch as `--revision`:

```bash
python -m tone_encoding.generate_classifier_input \
  --model_name techsword/wav2vec2-base-english-librispeech730h \
  --revision ckpt-5000 --dataset_name thchs30
```

`scripts/run_pretrain_pipeline.sh` uses this convention for all 18 checkpoints.
The loader (`tone_encoding.generate_classifier_input.loading_pretrained_model`)
imports the model as a torchaudio `Wav2Vec2Model`, so the 12-layer
`extract_features` path is unchanged. No manual download is needed; the Hub
fetches the branch on first use.

## Environment setup

`uv` manages the environment. Most dependencies are pinned in
`pyproject.toml`. The stack is torch 2.5.1, torchaudio 2.5.1, and transformers
4.46.3. Python 3.10, 3.11, and 3.12 are supported; torch 2.5.1 ships cp312
wheels. fairseq is no longer a dependency.

PyTorch is provided through two mutually exclusive extras. Select exactly one:

```bash
# CPU only (local development)
uv sync --extra cpu

# NVIDIA GPU with CUDA 12.1 (HPC)
uv sync --extra cu121
```

`uv sync` creates `.venv/` and installs the pinned dependency set. It also
installs the `tone_encoding` package from `src/` in
editable mode, so both `python -m tone_encoding.X` and
`from tone_encoding import X` work with no `sys.path` changes. The `cu121` extra
targets the validated CUDA 12.1 environment. Do not select both extras.

No `uv.lock` is committed. Torch resolution depends on the chosen accelerator
extra, so a committed lock would pin one backend for everyone.

## Exact torch-2.1.2 baseline

For comparisons that require the pre-2.5.1 numerical backend, use the immutable
tag `repro-torch-2.1.2`:

```bash
git checkout repro-torch-2.1.2
uv sync --extra cpu       # or --extra cu121
```

The tag preserves the pre-bump source and direct dependency pins: torch and
torchaudio 2.1.2, transformers 4.40.0, and Python 3.10 or 3.11. The matching
branch `repro/torch-2.1.2` exists for browsing. Use the tag as the canonical
reproduction reference.

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
`--contrast {tone,consonant}`, plus optional `--revision`, `--subclass`,
`--segment_input`, `--flattened`, `--cnn`, `--tgt_layers`, and `--seed` flags.
`--revision` selects a Hugging Face branch for the pretrained-checkpoint
experiments (see "Pretrained checkpoints" above).

## Legacy scripts

`tone_encoding.classification` is kept as a legacy predecessor of
`tone_encoding.experiment_classification`;
`tone_encoding.baselines_sanity_check` and `tone_encoding.plot_results` import
helper functions from it.

## License

Apache-2.0. See [LICENSE](LICENSE).
