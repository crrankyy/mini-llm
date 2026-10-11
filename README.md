# mini-llm

A GPT-style, decoder-only language model written from scratch in PyTorch.

The repository has three parts: a small library that defines the model and trains it (`src/mini_llm/`), the scripts that run it on WikiText-103 (`scripts/`), and the four study stages the library grew out of (`study/`).

## Status

This is a learning project, documented as it stands. Its known defects are described in [docs/08-known-issues.md](docs/08-known-issues.md).

## What is in here

| Path | Contents |
|---|---|
| `src/mini_llm/` | The library: `GPTModel` and its layers, dataloaders, the training loop, text generation, the model config |
| `scripts/prepare_fineweb.py` | Downloads and tokenizes the training data: 1 billion tokens, half FineWeb-Edu and half FineWeb |
| `scripts/prepare_wikitext.py` | Downloads WikiText-103, a much smaller dataset, and tokenizes it the same way |
| `scripts/train_wikitext.py` | Trains on those files, with a rolling checkpoint every 2,000 steps, per-epoch checkpoints and resume |
| `scripts/generate.py` | Loads trained weights and starts a prompt loop |
| `scripts/train_tiny.py` | Trains the same model on `data/basic-data.txt` |
| `train_colab.ipynb` | Runs the preparation and the training on a Colab GPU |
| `study/` | Stages 1 to 4 as standalone demos: tokenizers, sliding-window data, attention, the GPT building blocks |
| `tests/` | pytest tests |
| `data/` | `basic-data.txt`; the WikiText token files are generated here |
| `docs/` | Full documentation of everything above |

Checkpoints and weights are written to `checkpoints/`, which is not in git.

## The model

| Setting | Value |
|---|---|
| Tokenizer | `tiktoken`, `cl100k_base` |
| Vocabulary size | 100,352 |
| Context length | 1,024 tokens |
| Embedding dimension | 768 |
| Attention heads / layers | 12 / 12 |
| Dropout | 0 |
| Output head | shares the token embedding's matrix |
| Parameters | 162,885,120 (621 MB as float32) |

The config is named `GPT_CONFIG_124M` after GPT-2 small, but the larger vocabulary brings the model to 163M parameters. See [docs/04-model.md](docs/04-model.md#parameter-budget).

## Setup

Dependencies are declared in `pyproject.toml`. With [uv](https://docs.astral.sh/uv/):

```sh
uv sync                  # the library, its dependencies and pytest
uv sync --extra data     # also datasets and tqdm, which only the prepare scripts need
```

This installs `mini_llm` in editable mode, so the scripts can import it from any working directory. Without uv, `pip install -e ".[data]"` does the same.

On an Intel Mac the manifest pins `torch==2.2.2` and `numpy<2`. That is the last PyTorch release with wheels for the platform, and it needs Python 3.12 or older; `.python-version` selects 3.10. On other platforms any PyTorch from 2.2.2 on is accepted.

## Running

Run the scripts from the repo root, because the data and checkpoint paths are relative to the working directory.

```sh
uv run python scripts/prepare_fineweb.py     # writes data/fineweb-mix/train.bin and validation.bin, 4 GB
uv run python scripts/train_wikitext.py --data-dir data/fineweb-mix --epochs 1
uv run python scripts/generate.py            # interactive prompt
uv run pytest
```

- **Device.** The scripts use CUDA if it is available, then Apple's Metal backend (MPS), then the CPU.
- **Data.** `prepare_fineweb.py --tokens N` prepares a different amount and `--edu-share X` a different mix. Without `--data-dir`, `train_wikitext.py` reads `data/wikitext-103`, which `prepare_wikitext.py` writes.
- **Checkpoints.** `train_wikitext.py --checkpoint-dir DIR` writes somewhere other than `checkpoints/`; `--save-every N` and `--keep-checkpoints N` set how often the rolling checkpoint is saved and how many per-epoch checkpoints are kept. A run that stops, for any reason, is continued by running the script again. `generate.py --weights FILE` loads one specific file.
- **Training settings.** A batch is 4 windows of 1,024 tokens, and one weight update is made from 16 batches. The learning rate rises to 6e-4 over the first 3% of updates and then falls along a cosine. `--batch-size N`, `--accumulate N`, `--lr X` and `--warmup-share X` change these. Keep them, and `--epochs`, the same for the whole of a run; a run is not resumed with a different `--batch-size`.
- **Colab.** `train_colab.ipynb` runs the preparation and the training on a Colab GPU and keeps the checkpoints on Google Drive. See [docs/05-training.md](docs/05-training.md#colab-notebook).

`scripts/train_tiny.py` trains the full 163M-parameter model for 10 epochs and writes a checkpoint after each one, an estimated 19.5 GB in total. Read [docs/05-training.md](docs/05-training.md#train_tinypy) before running it.

Study scripts print their results and exit:

```sh
uv run python study/p01_tokenization/test_simple_tokenizer.py
uv run python study/p03_attention/self_attention.py
uv run python -m study.p02_data_loading.embed_dataloader
```

The command for every script, and what each one writes to disk, is in [docs/07-running.md](docs/07-running.md).

## Documentation

Start at [docs/README.md](docs/README.md). It has a repository map, a reference for every file, and links to the pages on [tokenization](docs/01-tokenization.md), the [data pipeline](docs/02-data-pipeline.md), [attention](docs/03-attention.md), the [model](docs/04-model.md), [training](docs/05-training.md), [inference](docs/06-inference.md), [running the code](docs/07-running.md) and [known issues](docs/08-known-issues.md).
