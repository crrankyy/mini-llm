# 07: Running the code

## Setup

Dependencies are declared in `pyproject.toml`, and `uv.lock` records the versions uv resolved.

| Package | Needed by | Declared as |
|---|---|---|
| `torch` | everything except the stage 1 regex tokenizers | dependency |
| `tiktoken` | every script that tokenizes with `cl100k_base` | dependency |
| `numpy` | `src/mini_llm/data.py`, `src/mini_llm/corpus.py`, `scripts/prepare_wikitext.py` | dependency |
| `datasets` | `scripts/prepare_fineweb.py`, `scripts/prepare_wikitext.py` | the `data` extra |
| `tqdm` | `scripts/prepare_wikitext.py` | the `data` extra |
| `pytest` | the tests | the `dev` group |

```sh
uv sync                  # the library, its dependencies and pytest
uv sync --extra data     # also datasets and tqdm
```

`uv sync` installs `mini_llm` in editable mode into `.venv/`. Without uv, `pip install -e ".[data]"` does the same; that is what the Colab notebook uses.

**Pins for the Intel Mac.** PyTorch 2.2.2 is the last release with wheels for Intel Macs. It was built against NumPy 1.x and has no wheel for Python 3.13 or later. The manifest therefore sets, for that platform only, `torch==2.2.2`, `numpy<2` and, in the `data` extra, `pyarrow<26` (pyarrow 26 needs NumPy 2). `.python-version` makes uv use Python 3.10. Other platforms get any PyTorch from 2.2.2 on.

Verified on the Intel Mac the docs were checked on:

| Command | Result |
|---|---|
| `uv sync` | Python 3.10.20, torch 2.2.2, numpy 1.26.4, tiktoken 0.14.0 |
| `uv sync --extra data` | adds datasets 5.1.0, pyarrow 25.0.1, pandas 2.3.3 |
| `pip install -e ".[data]"` in a clean Python 3.12 environment | same torch, numpy, datasets and pyarrow; the scripts ran from another directory |

Other platforms were not tried.

`tiktoken` downloads the `cl100k_base` vocabulary the first time it is used and caches it, so the first run needs network access.

## Run from the repo root

Imports no longer depend on the working directory: once the package is installed, `from mini_llm.model.gpt import GPTModel` resolves anywhere. Three things still do depend on it:

1. **Data paths.** `data/basic-data.txt` and `data/wikitext-103/` are opened relative to the working directory.
2. **Outputs.** `checkpoints/` is created relative to the working directory unless `--checkpoint-dir` says otherwise.
3. **Study scripts that import each other.** `study/` is not a package, so `import study.p02_data_loading.dataloader` resolves only when the repo root is on `sys.path`, which `python -m` arranges and running a file by path does not.

## Main workflow

```sh
uv run python scripts/prepare_fineweb.py     # downloads and tokenizes 1 billion tokens into data/fineweb-mix/
uv run python scripts/train_wikitext.py --data-dir data/fineweb-mix --epochs 1
                                             # trains, checkpoints every 2,000 steps, resumes if checkpoints exist
uv run python scripts/generate.py            # loads weights from checkpoints/, starts a prompt loop
```

`scripts/prepare_wikitext.py` prepares the much smaller WikiText-103 into `data/wikitext-103/`, which is where `train_wikitext.py` looks when `--data-dir` is not given.

| Option | Script | Effect |
|---|---|---|
| `--tokens N`, `--validation-tokens N`, `--edu-share X`, `--output-dir DIR` | `prepare_fineweb.py` | how much to prepare, how much to hold out, the share from FineWeb-Edu, and where to write (defaults: 1,000,000,000; 1,000,000; 0.5; `data/fineweb-mix`) |
| `--data-dir DIR` | `train_wikitext.py` | read `train.bin` and `validation.bin` from `DIR` (default `data/wikitext-103`) |
| `--epochs N` | `train_wikitext.py` | passes over the training data in total, across resumes (default 5) |
| `--checkpoint-dir DIR` | `train_wikitext.py` | write checkpoints and final weights to `DIR`, and resume from there |
| `--save-every N` | `train_wikitext.py` | save the rolling checkpoint every `N` training steps (default 2000) |
| `--keep-checkpoints N` | `train_wikitext.py` | keep the newest `N` end-of-epoch checkpoints and delete older ones (default 10) |
| `--batch-size N` | `train_wikitext.py` | `N` windows of 1,024 tokens in one batch (default 4) |
| `--accumulate N` | `train_wikitext.py` | add up the gradients of `N` batches for one weight update (default 16) |
| `--lr X`, `--warmup-share X` | `train_wikitext.py` | the peak learning rate, and the share of all weight updates during which the rate rises to it (defaults: 6e-4; 0.03) |
| `--checkpoint-dir DIR` | `generate.py` | search `DIR` for weights |
| `--weights FILE` | `generate.py` | load exactly this weights or checkpoint file |

And the small-data variant:

```sh
uv run python scripts/train_tiny.py
```

Before running `train_tiny.py`, note what it does: it trains the full 163M-parameter model for 10 epochs and writes a checkpoint after each epoch into `checkpoints/tiny`, about 19.5 GB in total. See [05-training.md](05-training.md#train_tinypy).

How far each of these was run for the docs is listed in [README.md](README.md#snapshot). `train_wikitext.py` does not run on macOS as written; see [known issue 12](08-known-issues.md#12-train_wikitextpy-cannot-start-its-dataloader-workers-on-macos). The intended way to train is `train_colab.ipynb`, described in [05-training.md](05-training.md#colab-notebook).

## Study scripts

Each command was run from the repo root, with the project environment active or through `uv run`.

| Command | Result |
|---|---|
| `python study/p01_tokenization/simple_tokenizer.py` | runs; ends with "Decoded text matches the original text!" |
| `python study/p01_tokenization/test_simple_tokenizer.py` | runs |
| `python study/p01_tokenization/test_byte_pair_encoding.py` | runs |
| `python study/p02_data_loading/dataloader.py` | runs |
| `python study/p02_data_loading/embedding.py` | runs |
| `python -m study.p02_data_loading.embed_dataloader` | runs |
| `python study/p02_data_loading/embed_dataloader.py` | fails: `ModuleNotFoundError: No module named 'study'` |
| `python study/p03_attention/simple_attention.py` | runs |
| `python study/p03_attention/self_attention.py` | runs |
| `python study/p04_gpt/architecture.py` | runs |
| `python -m study.p04_gpt.architecture` | runs |

Except for `embed_dataloader`, whose embedding is not seeded, and `architecture.py`, which could not start before, every script prints exactly what it printed before the reorganization (verified by comparing the output).

The modules under `src/mini_llm/` only define things; importing them prints nothing.

## Tests

```sh
uv run pytest
```

Verified: `43 passed`. pytest is configured in `pyproject.toml` to collect from `tests/` only, because `study/` contains demo scripts named `test_*.py` that run at import.

## Devices

`train_tiny.py`, `train_wikitext.py` and `generate.py` all call `get_device()` from `src/mini_llm/device.py`, which returns `cuda` if it is available, then `mps` (Apple's Metal backend), then `cpu`.

| | CUDA | MPS | CPU |
|---|---|---|---|
| Mixed precision and gradient scaler | yes | no | no |
| `torch.compile` in `train_wikitext.py` | attempted | skipped | attempted |

Measured on the Intel MacBook the docs were checked on (Radeon Pro 5300M, 4 GB), with torch 2.2.2:

| Workload, full 163M model | CPU | MPS |
|---|---|---|
| One forward pass, 1 x 1,024 tokens | 0.920 s | 0.246 s |
| One training step, batch 1 x 1,024 | 4.28 s | 1.98 s |
| One training step, batch 2 x 1,024 | 7.68 s | out of memory |

The MPS training step is the median of six; the fastest took 1.79 s.

So MPS is worth having for `generate.py`. For training, the best that machine does is about 520 tokens a second, at which one pass over the FineWeb mix would take about 22 days. The batch of 4 x 1,024 that `train_wikitext.py` uses does not fit its GPU.

With the 256-token context, before 2026-10-12, a forward pass over 1 x 256 tokens took 0.242 s and 0.071 s, a training step at 2 x 256 took 2.78 s and 1.39 s, a step at 4 x 256 took 1.72 s on MPS, and 8 x 256 ran out of memory there.

Before the model changed on 2026-10-11 (239M parameters, hand-written attention) the same measurements were 0.300 s and 0.093 s for the forward pass, 3.77 s and 2.68 s for the training step at batch 2, and batch 4 ran out of memory on MPS.

## What gets written to disk

| Path | Written by | Contents | In `.gitignore` |
|---|---|---|---|
| `data/fineweb-mix/train.bin`, `validation.bin`, `manifest.json` | `prepare_fineweb.py` | flat `uint32` token ids, 4 bytes each, no header, with token 100,257 after every document; the manifest records what was prepared | yes |
| `data/wikitext-103/train.bin`, `validation.bin` | `prepare_wikitext.py` | flat `uint32` token ids, 4 bytes each, no header | yes |
| `checkpoints/gpt_wikitext_checkpoint_latest_a.pth`, `..._latest_b.pth` | the trainer, called by `train_wikitext.py` | the rolling checkpoint, two files written in turn. Dict: `epoch`, `batches_done`, `model_state_dict`, `optimizer_state_dict`, `scaler_state_dict`, `global_step`, `tokens_seen`, `updates_done` | yes |
| `checkpoints/gpt_wikitext_checkpoint_ep{N}.pth` | the trainer, called by `train_wikitext.py` | the same dict at the end of epoch `N`; the newest 10 are kept | yes |
| `checkpoints/tiny/gpt_wikitext_checkpoint_ep{N}.pth` | the trainer, called by `train_tiny.py` | the same dict | yes |
| `checkpoints/gpt_wikitext_124M.pth` | `train_wikitext.py` at exit | bare state dict; keys prefixed `_orig_mod.` if the model was compiled | yes |
| `/content/data/<folder name>/` | `train_wikitext.py` on Colab, only when the data folder is on Google Drive | copy of the `.bin` files | outside the repo |
| `.venv/` | `uv sync` | the project environment | yes |
| `__pycache__/` | Python | bytecode | yes |

With `--checkpoint-dir`, the three `train_wikitext.py` rows move to that folder. The checkpoint files are described in [05-training.md](05-training.md#checkpoints). Hugging Face `datasets` also keeps its own download cache outside the repo.
