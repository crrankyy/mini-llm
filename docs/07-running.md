# 07: Running the code

## Environment

The repository declares no dependencies: there is no `pyproject.toml`, `requirements.txt` or lock file. The imports are:

| Package | Needed by |
|---|---|
| `torch` | everything except the stage 1 regex tokenizers |
| `tiktoken` | every script that tokenizes with `cl100k_base` |
| `numpy` | `src/utils/dataloader.py`, `prepare_wikitext.py` |
| `datasets`, `tqdm` | `prepare_wikitext.py` only |

The checks in these docs ran in a throwaway environment that leaves the repo untouched:

```sh
uv run --no-project --python 3.10 --with 'torch==2.2.2' --with 'numpy<2' --with tiktoken python <script>
```

Those pins are what was used on the Intel Mac the checks ran on. Other versions were not tried.

`tiktoken` downloads the `cl100k_base` vocabulary the first time it is used and caches it, so the first run needs network access.

## Always run from the repo root

Three things depend on it:

1. **Imports.** There are no `__init__.py` files and nothing is installed as a package. `from src.model import GPTModel` resolves only when the repo root is on `sys.path`.
2. **Data paths.** `data/basic-data.txt` and `data/wikitext-103/` are opened relative to the working directory.
3. **Outputs.** Checkpoints and weights are written to the working directory.

Python puts a script's own folder on `sys.path` when the script is run by path, and the working directory when a module is run with `-m`. That decides how each file has to be started:

- Root scripts (`test.py`, `train_wikitext.py`, ...) run by path, because their folder is the root.
- Scripts under `src/` that import `src.*` must be run with `-m`.
- Scripts under `src/` that import only third-party packages, or a sibling by bare name, run by path.

## Study scripts

Each command was run from the repo root.

| Command | Result |
|---|---|
| `python device.py` | runs |
| `python src/p01_tokenization/simple_tokenizer.py` | runs; ends with "Decoded text matches the original text!" |
| `python src/p01_tokenization/test_simple_tokenizer.py` | runs |
| `python src/p01_tokenization/test_byte_pair_encoding.py` | runs |
| `python src/p02_data_loading/dataloader.py` | runs |
| `python src/p02_data_loading/embedding.py` | runs |
| `python -m src.p02_data_loading.embed_dataloader` | runs |
| `python src/p02_data_loading/embed_dataloader.py` | fails: `ModuleNotFoundError: No module named 'src'` |
| `python src/p03_attention/simple_attention.py` | runs |
| `python src/p03_attention/self_attention.py` | runs |
| `python src/p04_gpt/architecture.py` | fails: `ModuleNotFoundError: No module named 'sfeedforward'` |
| `python -m src.p04_gpt.architecture` | fails with the same error |

`attention_modules.py`, `feedforward.py`, `loss.py`, `textgen.py` and the files in `src/utils/` only define things; running them prints nothing.

## Main workflow

None of these four was run for these docs.

```sh
python prepare_wikitext.py    # downloads WikiText-103, writes data/wikitext-103/*.bin
python train_wikitext.py      # trains, checkpoints each epoch, resumes if checkpoints exist
python test_inference.py      # loads weights from the working directory, starts a prompt loop
```

And the small-data variant:

```sh
python test.py                # CPU only
```

Before running `test.py`, note what it does: it trains the full 239M-parameter model for 10 epochs on CPU and writes a checkpoint after each epoch, an estimated 29 GB in total. See [05-training.md](05-training.md#testpy).

Before running `train_wikitext.py` or `test_inference.py`, check which `.pth` files are in the working directory. Both scripts choose files by name pattern; see [known issues 3 and 6](08-known-issues.md).

## Devices

| Script | Device |
|---|---|
| `test.py` | `cpu`, hardcoded |
| `train_wikitext.py`, `test_inference.py` | `cuda` if available, otherwise `cpu` |

There is no MPS branch anywhere, so on a Mac everything runs on CPU. `device.py` prints the selected device but no script imports it.

Mixed precision and the gradient scaler are enabled only on CUDA.

## What gets written to disk

| Path | Written by | Contents | In `.gitignore` |
|---|---|---|---|
| `data/wikitext-103/train.bin`, `validation.bin` | `prepare_wikitext.py` | flat `uint32` token ids, 4 bytes each, no header | no |
| `gpt_wikitext_checkpoint_ep{N}.pth` | the trainer, so both `test.py` and `train_wikitext.py` | dict: `epoch`, `model_state_dict`, `optimizer_state_dict`, `global_step`, `tokens_seen` | no |
| `gpt_wikitext_checkpoint_interrupted.pth` | `train_wikitext.py` on `Ctrl-C` | dict: `epoch`, `model_state_dict`, `optimizer_state_dict` | no |
| `gpt_wikitext_124M.pth` | `train_wikitext.py` at exit | bare state dict; keys prefixed `_orig_mod.` if the model was compiled | no |
| `/content/data/wikitext-103/` | `train_wikitext.py` on Colab | copy of the `.bin` files | outside the repo |
| `__pycache__/` | Python | bytecode | yes |

Hugging Face `datasets` also keeps its own download cache outside the repo.

Since none of the `.pth` or `.bin` outputs are ignored, they show up in `git status` after a run.
