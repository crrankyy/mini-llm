# mini-llm

A GPT-style, decoder-only language model written from scratch in PyTorch.

The repository builds the pieces up in four stages (tokenization, data loading, attention, the GPT architecture), then wires them into scripts that train the model on WikiText-103 and sample from it at an interactive prompt.

## Status

This is a learning project, documented as it stands. Its nine known defects are described in [docs/08-known-issues.md](docs/08-known-issues.md).

## What is in here

| Path | Contents |
|---|---|
| `src/p01_tokenization/` | Stage 1: a regex tokenizer, then the `tiktoken` BPE tokenizer |
| `src/p02_data_loading/` | Stage 2: sliding-window datasets and embedding lookups |
| `src/p03_attention/` | Stage 3: attention, from plain dot products to multi-head |
| `src/p04_gpt/` | Stage 4: layer norm, GELU, feed-forward, and a sandbox script |
| `src/model.py`, `src/transformer.py` | `GPTModel` and `TransformerBlock` |
| `src/utils/` | Dataloaders, loss, evaluation, the training loop, text generation |
| `config.py` | The model config |
| `prepare_wikitext.py` | Downloads WikiText-103 and tokenizes it into binary files |
| `train_wikitext.py` | Trains on those files, with per-epoch checkpoints and resume |
| `test_inference.py` | Loads trained weights and starts a prompt loop |
| `test.py` | Trains the same model on `data/basic-data.txt` on CPU |
| `docs/` | Full documentation of everything above |

## The model

| Setting | Value |
|---|---|
| Tokenizer | `tiktoken`, `cl100k_base` |
| Vocabulary size | 100,352 |
| Context length | 256 tokens |
| Embedding dimension | 768 |
| Attention heads / layers | 12 / 12 |
| Dropout | 0.1 |
| Parameters | 239,365,632 (913 MB as float32) |

The config is named `GPT_CONFIG_124M` after GPT-2 small, but the larger vocabulary and the untied output head bring the model to 239M parameters. See [docs/04-model.md](docs/04-model.md#parameter-budget).

## Requirements

The repository has no dependency manifest. The code imports:

- `torch`, `tiktoken`, `numpy`
- `datasets` and `tqdm`, for `prepare_wikitext.py` only

One way to run a script without installing anything into the project, using [uv](https://docs.astral.sh/uv/):

```sh
uv run --no-project --python 3.10 --with 'torch==2.2.2' --with 'numpy<2' --with tiktoken python src/p03_attention/simple_attention.py
```

Those pins are the ones the documentation was checked with, on an Intel Mac. Other versions were not tried.

## Running

Run everything from the repo root. Imports and data paths only resolve from there.

Study scripts print their results and exit:

```sh
python src/p01_tokenization/test_simple_tokenizer.py
python src/p03_attention/self_attention.py
python -m src.p02_data_loading.embed_dataloader
```

Training on WikiText-103 and sampling from the result:

```sh
python prepare_wikitext.py    # writes data/wikitext-103/train.bin and validation.bin
python train_wikitext.py      # uses CUDA if available; checkpoints each epoch
python test_inference.py      # interactive prompt
```

`test.py` trains the full 239M-parameter model for 10 epochs on CPU and writes a checkpoint after each epoch, an estimated 29 GB in total. Read [docs/05-training.md](docs/05-training.md#testpy) before running it.

The command for every script, and what each one writes to disk, is in [docs/07-running.md](docs/07-running.md).

## Documentation

Start at [docs/README.md](docs/README.md). It has a repository map, a reference for every file, and links to the pages on [tokenization](docs/01-tokenization.md), the [data pipeline](docs/02-data-pipeline.md), [attention](docs/03-attention.md), the [model](docs/04-model.md), [training](docs/05-training.md), [inference](docs/06-inference.md), [running the code](docs/07-running.md) and [known issues](docs/08-known-issues.md).
