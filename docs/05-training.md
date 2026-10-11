# 05: Training

Training is next-token prediction: the model's logits at each position are compared with the token that follows it, using cross-entropy. The shared pieces are in `src/mini_llm/training/`; `scripts/train_tiny.py` and `scripts/train_wikitext.py` are the two scripts that call them, and `train_colab.ipynb` runs the second one on Colab.

## Loss (`src/mini_llm/training/loss.py`)

### `calc_loss_batch(input_batch, target_batch, model, device)`

Moves both tensors to `device`, runs the model, and returns the mean cross-entropy over all `B x T` positions:

```python
logits = model(input_batch)
loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
```

`logits.flatten(0, 1)` is `(B*T, V)` and `target_batch.flatten()` is `(B*T,)`. The returned loss is a tensor that still carries its graph, so the caller can call `backward()` on it.

### `calc_loss_loader(data_loader, model, device, num_batches=None)`

Averages `calc_loss_batch` over the first `num_batches` batches of a loader, or over all of them if `num_batches` is `None`. It returns `nan` for an empty loader and a Python float otherwise. It does not switch the model to eval mode or disable gradients; the caller does that.

## Evaluation (`src/mini_llm/training/evaluate.py`)

`evaluate_model(model, train_loader, val_loader, device, eval_iter)` puts the model in eval mode, computes `calc_loss_loader` on `eval_iter` batches from each loader under `torch.no_grad()`, puts the model back in train mode, and returns `(train_loss, val_loss)`.

The training loader is shuffled, so each call scores a different random set of training batches.

## The training loop (`src/mini_llm/training/trainer.py`)

```python
train_model_v1(model, train_loader, val_loader, optimizer, device, num_epochs,
               eval_frq, eval_iter, start_context, tokenizer, start_epoch=0,
               checkpoint_dir="checkpoints")
```

Returns `(train_losses, val_losses, track_tokens_seen)`, three lists with one entry per evaluation.

It creates `checkpoint_dir` before the first batch, so a path that cannot be created fails at the start and not after an epoch.

Per batch:

| Device | Steps |
|---|---|
| CUDA | `zero_grad`; forward and loss inside `torch.autocast(device_type='cuda', dtype=torch.float16)`; `scaler.scale(loss).backward()`; `scaler.unscale_(optimizer)`; clip gradient norm to 1.0; `scaler.step(optimizer)`; `scaler.update()` |
| CPU and MPS | `zero_grad`; forward and loss; `loss.backward()`; clip gradient norm to 1.0; `optimizer.step()` |

The CUDA path uses mixed precision with a `torch.cuda.amp.GradScaler`. Gradients are unscaled before clipping so that the 1.0 threshold applies to their true size. The CUDA path was not run for these docs. MPS takes the second path in full precision; torch 2.2.2 has no autocast for it.

After each batch, `tokens_seen` grows by `input_batch.numel()` and `global_step` by one. `global_step` starts at -1, so the first batch is step 0.

**Every `eval_frq` steps**, starting with step 0, it calls `evaluate_model`, appends to the three lists, and prints:

```text
Ep 1 (Step 000005) [2026-10-10 23:06:17]: Train loss 11.657, Val loss 11.665
```

**At the end of every epoch** it prints a 50-token sample from `start_context` (see [06-inference.md](06-inference.md#generate_and_print_samplemodel-tokenizer-device-start_context)) and then saves a checkpoint. The order matters: if printing the sample fails, that epoch's checkpoint is not written. The one known way for it to fail, a sampled id the tokenizer cannot decode, is fixed; see [known issue 11](08-known-issues.md#11-sampling-an-id-the-tokenizer-cannot-decode-stopped-the-run-fixed).

There is no learning-rate schedule and no gradient accumulation. The learning rate is whatever the optimizer was built with.

### Checkpoints

Written to `checkpoint_dir` as `gpt_wikitext_checkpoint_ep{N}.pth`, where `N` is the 1-based epoch number. The file name is hardcoded in the trainer, so `train_tiny.py` writes files with this name too, into its own folder.

| Key | Value |
|---|---|
| `epoch` | number of completed epochs |
| `model_state_dict` | `model.state_dict()` |
| `optimizer_state_dict` | `optimizer.state_dict()` |
| `global_step` | last step index of this call |
| `tokens_seen` | tokens processed in this call |

`global_step` and `tokens_seen` are saved but never read back. Each call to `train_model_v1` starts both from scratch, so after a resume the step counter in the log restarts at 0 (verified).

**Size.** A checkpoint of the full model is 2.88 GB (measured), three times the 957 MB of float32 weights, because AdamW keeps two extra tensors per parameter.

### Verified run

`train_wikitext.py` was run with the model shrunk to one 8-dimensional layer and a 32-token context, small stand-in token files, and its DataLoader workers switched off, on CPU and again on MPS:

- A first run trained five epochs and wrote `..._ep1.pth` to `..._ep5.pth` and the final weights.
- A second run found `ep5`, printed "Already completed 5 epochs", and trained nothing.
- With `ep4` and `ep5` deleted, a third run resumed from `ep3`, logged `Ep 4 (Step 000000)`, and wrote `ep4` and `ep5` again.

The starting loss was 11.66, close to the 11.5 that a model guessing uniformly over the vocabulary would score.

A second set of runs forced the CPU, where the script compiles the model, and shared one checkpoint folder. `torch.compile` was switched to its `eager` backend, which still returns the `OptimizedModule` wrapper; the code its default backend generates hangs on the machine these docs were checked on.

- A first run was killed in the middle of epoch 3. It left `ep1` and `ep2`.
- A second run resumed from `ep2` into the compiled model, with every weight tensor and the whole optimizer state equal to the checkpoint. It was interrupted in epoch 4 and left `ep3`, the interrupted checkpoint and the final-weights file.
- A third run resumed from `ep3`, not from the interrupted checkpoint, and wrote `ep4` and `ep5`.
- `generate.py` loaded both the final weights and `ep3`.

## `train_tiny.py`

`scripts/train_tiny.py` is a module-level script. It was run for these docs only with the shrunken model described above, because as written it trains the full 239M-parameter model. In order, it:

1. Picks the device with `get_device()`.
2. Builds the tokenizer and stacks two four-token sentences into a tensor named `batch`, which is not used afterwards.
3. Seeds with 123, builds `GPTModel(GPT_CONFIG_124M)`, and prints the parameter count and size.
4. Generates 10 tokens greedily from "Every effort moves you" with the untrained model.
5. Reads `data/basic-data.txt`, prints its character and token counts, and splits it 90/10 **by characters**.
6. Builds the two loaders (see [02-data-pipeline.md](02-data-pipeline.md#which-loader-each-script-uses)), prints every batch shape, and prints the initial train and validation loss.
7. Builds a **second, new** `GPTModel` and an `AdamW` optimizer, and trains it.

| Setting | Value |
|---|---|
| Optimizer | `AdamW(lr=0.0004, weight_decay=0.1)` |
| Epochs | 10 |
| `eval_frq` / `eval_iter` | 5 / 5 |
| Batches per epoch | 8 |
| Sample prompt | "Every effort moves you" |
| Checkpoint folder | `checkpoints/tiny` |

It leaves ten `gpt_wikitext_checkpoint_ep*.pth` files in `checkpoints/tiny`, about 29 GB at 2.88 GB each. Because that is a separate folder, `train_wikitext.py` and `generate.py` no longer pick these files up.

## `train_wikitext.py`

`scripts/train_wikitext.py` is a module-level script. It was run end to end only as described under [Verified run](#verified-run): not at full size, not on CUDA, and with `torch.compile` only through its `eager` backend. On macOS it cannot run as written; see [known issue 12](08-known-issues.md#12-train_wikitextpy-cannot-start-its-dataloader-workers-on-macos).

```sh
uv run python scripts/train_wikitext.py                          # checkpoints/ under the working directory
uv run python scripts/train_wikitext.py --checkpoint-dir DIR     # for example a folder on Google Drive
```

In order, it:

1. Reads `--checkpoint-dir` (default `checkpoints`).
2. Picks the device with `get_device()`. Seeds with 123, builds the model, moves it to the device, prints its size.
3. Sets `data_dir = "data/wikitext-103"`. If `/content` exists, which is how it detects Colab, it copies that folder to `/content/data/wikitext-103` once and reads from there.
4. Builds the memory-mapped loaders and prints their batch counts.
5. Builds `AdamW(lr=5e-4, weight_decay=0.1)`.
6. Looks in the checkpoint folder for a checkpoint to resume from, and loads it.
7. Unless the device is MPS, calls `torch.compile(model)` inside a `try`; if that raises, it prints a message and continues uncompiled.
8. Trains the remaining epochs, or prints that training is already complete.
9. Saves the final weights into the checkpoint folder.

Compilation is skipped on MPS because there `torch.compile` returns without error and the first forward pass then fails with `Device mps not supported` (verified on torch 2.2.2), outside the `try`. It comes after the resume because the compiled wrapper cannot load the checkpoint; see [Resume](#resume).

| Setting | Value |
|---|---|
| Batch size | 16 |
| Window / stride | 256 / 256 |
| `num_workers` | 2 |
| Optimizer | `AdamW(lr=5e-4, weight_decay=0.1)` |
| Epochs (`num_epochs`) | 5 in total, across resumes |
| `eval_frq` / `eval_iter` | 500 / 10 |
| Sample prompt | "The secret to training large models is" |

With the real `train.bin` of 120,196,189 tokens, the training loader has 29,344 batches per epoch and the validation loader 62 (verified by building both loaders). Five epochs are 146,720 steps, about 601 million tokens.

### Resume

```python
checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "gpt_wikitext_checkpoint_ep*.pth")))
if checkpoints:
    latest_checkpoint = checkpoints[-1]
    checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    unwrapped_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(unwrapped_state_dict)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
```

`torch.compile` wraps the model in an `OptimizedModule` whose state-dict keys start with `_orig_mod.`, so a checkpoint saved from a compiled model carries that prefix. The code strips it and loads the result into the plain model, which is compiled only afterwards. Loading into the wrapper fails; that was [known issue 2](08-known-issues.md#2-resuming-into-a-compiled-model-failed-fixed).

- **Granularity is one epoch.** Training restarts at the beginning of the first unfinished epoch.
- **The latest file is chosen by string sort.** `ep10` sorts before `ep2`, so this picks the wrong file once there are ten or more. With `num_epochs = 5` that does not happen.
- **A run with a compiled model can be resumed.** Verified on torch 2.2.2 as described under [Verified run](#verified-run). Not checked on the torch version Colab installs.
- **A checkpoint that was cut off stops the run.** The newest file is loaded with no fallback. Verified by truncating the newest checkpoint: the next run stopped with `RuntimeError: PytorchStreamReader failed reading zip archive` and left the older checkpoints alone. Deleting the damaged file lets the run resume from the one before it.

### Interrupt and final save

`Ctrl-C` during training is caught. The handler saves `gpt_wikitext_checkpoint_interrupted.pth` in the checkpoint folder with `epoch` set to the epoch the run started from, and prints that the next run will resume from it. It will not: the resume pattern is `..._ep*.pth`, which this name does not match.

After the `try` block the script saves `model.state_dict()` to `gpt_wikitext_124M.pth` in the checkpoint folder. That includes the case where training was interrupted and the case where nothing was trained because all epochs were already done. When the model was compiled, the keys in this file carry the `_orig_mod.` prefix; `generate.py` strips it on load.

### Files it reads and writes

| Path | Direction | When |
|---|---|---|
| `data/wikitext-103/train.bin`, `validation.bin` | read | always |
| `/content/data/wikitext-103/` | written | on Colab, first run |
| `<checkpoint dir>/gpt_wikitext_checkpoint_ep{N}.pth` | read and written | resume; end of each epoch |
| `<checkpoint dir>/gpt_wikitext_checkpoint_interrupted.pth` | written | on `Ctrl-C` |
| `<checkpoint dir>/gpt_wikitext_124M.pth` | written | at exit |

## Colab notebook

`train_colab.ipynb` sits at the repo root and is meant to be opened in VS Code with the Colab extension, so the file stays local while its cells run on a Colab GPU. Nothing it produces lands on the local machine.

| Step | What the cell does |
|---|---|
| Settings and helpers | Defines the paths and the helper functions. Safe to re-run at any time; it starts nothing. |
| 1. Check the GPU | Stops if the kernel has no CUDA device. |
| 2. Mount Google Drive | Mounts Drive at `/content/drive` and stops if the mount is missing, since the checkpoints would otherwise go to the temporary disk. |
| 3. Get the code | Clones the private repo into `/content/mini-llm` with a GitHub token typed at a prompt. The token is passed to git through the environment and is not stored. |
| 4. Install | `pip install -e` of the clone with the `data` extra. |
| 5. Prepare the data | Runs `prepare_wikitext.py` unless the token files already exist. |
| 6. Start training | Starts `train_wikitext.py --checkpoint-dir <Drive folder>` as a detached background process, with its output appended to a log file on Drive. Does nothing if training is already running. |
| 7. Watch progress | Parses the evaluation lines out of the log and plots train and validation loss; a second cell shows the end of the log and the checkpoint folder. |
| 8. Stop training | Sends the interrupt that `Ctrl-C` would send. |

Where things live during a Colab run:

| What | Where | Kept after the session |
|---|---|---|
| The clone, the installed package | `/content/mini-llm` | no |
| `train.bin`, `validation.bin` | `/content/mini-llm/data/wikitext-103/`, copied to `/content/data/wikitext-103/` | no; rebuilt by step 5 |
| Checkpoints and final weights | `mini-llm/checkpoints/` on Google Drive | yes |
| The training log | `mini-llm/train.log` on Google Drive | yes |

To sample from a checkpoint locally, download it from Drive into `checkpoints/` and pass it to `generate.py --weights`; see [06-inference.md](06-inference.md#generatepy).

Verified locally: the clone cell against the real repository with a valid and an invalid token; the start, status and stop cells against a stand-in script that forks two workers; and the progress cell against a log written by the real trainer, where it parsed 138 of 138 evaluation lines. **Not run:** the notebook on Colab itself, so the GPU check, the Drive mount, the install and the data preparation are untested there.

### Across sessions

A new session resumes from the newest epoch checkpoint on Drive. Three limits to know before a long run:

- **Only finished epochs are kept.** A checkpoint is written at the end of an epoch, so whatever a session trained after its last finished epoch is lost when the session ends. The stop cell does not change that: the file it makes the script save is not one a later run resumes from ([known issue 3](08-known-issues.md#3-the-interrupt-checkpoint-is-never-resumed-and-final-weights-are-saved-regardless)). An epoch, 29,344 steps, therefore has to fit inside one session. How long a step takes on a Colab GPU was not measured; the log gets a time-stamped line every 500 steps, so the first minutes of a run show it.
- **Space on Drive.** Five checkpoints of 2.88 GB and the 957 MB of final weights come to about 15.4 GB, and using the stop cell adds one more file the size of a checkpoint.
- **A checkpoint that was cut off** while it was being written stops the next session until the file is deleted; see [Resume](#resume).
