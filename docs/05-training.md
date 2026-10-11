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
               checkpoint_dir="checkpoints", save_frq=None, keep_checkpoints=10,
               resume=None, eval_train_loader=None)
```

| Parameter | Effect |
|---|---|
| `save_frq` | Steps between saves of the rolling checkpoint. `None`, the default, switches the rolling checkpoint off; `train_tiny.py` leaves it off. |
| `keep_checkpoints` | How many end-of-epoch checkpoints to keep. Older ones are deleted each time a new one is written. `None` keeps them all. |
| `resume` | The position to continue from, as `training_progress` returns it for a loaded checkpoint. It restores the step and token counters, the number of batches already done in the current epoch, and the gradient scaler. The caller loads the weights and the optimizer state itself. |
| `eval_train_loader` | The loader evaluation takes its training batches from. Defaults to `train_loader`. |

Returns `(train_losses, val_losses, track_tokens_seen)`, three lists with one entry per evaluation.

It creates `checkpoint_dir` before the first batch, so a path that cannot be created fails at the start and not after an epoch.

Per batch:

| Device | Steps |
|---|---|
| CUDA | `zero_grad`; forward and loss inside `torch.autocast(device_type='cuda', dtype=torch.float16)`; `scaler.scale(loss).backward()`; `scaler.unscale_(optimizer)`; clip gradient norm to 1.0; `scaler.step(optimizer)`; `scaler.update()` |
| CPU and MPS | `zero_grad`; forward and loss; `loss.backward()`; clip gradient norm to 1.0; `optimizer.step()` |

The CUDA path uses mixed precision with a `torch.cuda.amp.GradScaler`. Gradients are unscaled before clipping so that the 1.0 threshold applies to their true size. The CUDA path was not run for these docs. MPS takes the second path in full precision; torch 2.2.2 has no autocast for it.

After each batch, `tokens_seen` grows by `input_batch.numel()` and `global_step` by one. `global_step` starts at -1, so the first batch is step 0. A third counter, `batches_done`, counts the batches finished in the current epoch. After a resume all three continue from the checkpoint.

**Every `eval_frq` steps**, starting with step 0, it calls `evaluate_model`, appends to the three lists, and prints:

```text
Ep 1 (Step 000005) [2026-10-10 23:06:17]: Train loss 11.657, Val loss 11.665
```

**Every `save_frq` steps** it saves the rolling checkpoint, described under [Checkpoints](#checkpoints).

**At the end of every epoch** it prints a 50-token sample from `start_context` (see [06-inference.md](06-inference.md#generate_and_print_samplemodel-tokenizer-device-start_context)), saves the end-of-epoch checkpoint, deletes the end-of-epoch checkpoints beyond `keep_checkpoints`, and saves the rolling checkpoint again. The order matters: if printing the sample fails, that epoch's checkpoint is not written, and a later run continues from the last rolling save. The one known way for it to fail, a sampled id the tokenizer cannot decode, is fixed; see [known issue 11](08-known-issues.md#11-sampling-an-id-the-tokenizer-cannot-decode-stopped-the-run-fixed).

There is no learning-rate schedule and no gradient accumulation. The learning rate is whatever the optimizer was built with.

### Checkpoints

The trainer writes two kinds of file into `checkpoint_dir`. The names are fixed in `src/mini_llm/training/checkpoint.py`, so `train_tiny.py` writes end-of-epoch files with the same name, into its own folder.

| File | Written | Purpose |
|---|---|---|
| `gpt_wikitext_checkpoint_ep{N}.pth` | at the end of epoch `N`, counted from 1 | one snapshot per epoch; the newest `keep_checkpoints` are kept |
| `gpt_wikitext_checkpoint_latest_a.pth`, `gpt_wikitext_checkpoint_latest_b.pth` | every `save_frq` steps, at the end of every epoch, and on `Ctrl-C`; not at all when `save_frq` is `None` | the rolling checkpoint, which a later run resumes from |

Both kinds hold the same dict:

| Key | Value |
|---|---|
| `epoch` | number of completed epochs |
| `batches_done` | batches finished in the epoch after those; 0 in an end-of-epoch checkpoint |
| `model_state_dict` | `model.state_dict()` |
| `optimizer_state_dict` | `optimizer.state_dict()` |
| `scaler_state_dict` | the gradient scaler's state on CUDA, otherwise `None` |
| `global_step` | index of the last finished step |
| `tokens_seen` | tokens processed so far |

Checkpoints written before the rolling checkpoint existed have neither `batches_done` nor `scaler_state_dict`. They still load, as a position at the start of an epoch (verified).

**The rolling checkpoint is two files written in turn.** Each save overwrites the older file in place, so a save that is cut off damages one file and leaves the other. The first save of a run never goes to the file the run resumed from, which until then is the only good copy. Nothing is renamed or deleted: on Google Drive a deleted file goes to the Trash and keeps counting against the storage quota, and whether a rename over an existing file does the same could not be established.

**On `Ctrl-C`** the trainer saves the position it has reached to the rolling checkpoint and raises the interrupt again. If a save was in progress, it is redone into the same file. An interrupt can also arrive in the middle of an optimizer update; the state saved then holds whatever part of that update had been applied, and its batch can be counted one off.

**Pruning.** After each end-of-epoch checkpoint, the ones beyond the newest `keep_checkpoints` are deleted with `os.remove`. Only files named `gpt_wikitext_checkpoint_ep{N}.pth` in `checkpoint_dir` are touched, and they are ordered by `N` as a number.

**Continuing inside an epoch.** `train_wikitext.py` builds its training loader with `seed=123`, which gives the loader a `ResumableSampler` (see [02-data-pipeline.md](02-data-pipeline.md#resumablesamplernum_samples-seed)). The order of an epoch then depends only on the seed and the epoch number. The trainer starts each epoch with `sampler.set_position(epoch, batches_done * batch_size)`, so a resumed run visits exactly the windows the stopped run had not reached. With an ordinary loader a position inside an epoch cannot be restored, and the trainer raises `ValueError`. Evaluation takes its training batches from `eval_train_loader`, an ordinary shuffled loader, so they remain a different random set at every evaluation.

**Size.** A checkpoint of the full model is 2.88 GB (measured), three times the 957 MB of float32 weights, because AdamW keeps two extra tensors per parameter. On the Intel MacBook's internal disk one rolling save of the full model took 6 to 7 seconds and loading it back 3.6 seconds (measured). A run holds two rolling files, up to `keep_checkpoints` end-of-epoch files, and the final weights.

### Verified run

`train_wikitext.py` was run with the model shrunk to one 8-dimensional layer and a 32-token context, small stand-in token files with 6 batches per epoch, and its DataLoader workers switched off.

On MPS, where the model is not compiled:

- A first run trained five epochs and wrote `..._ep1.pth` to `..._ep5.pth`, the two rolling files and the final weights.
- A second run found the newer rolling file, printed "Already completed 5 epochs", and trained nothing.

The starting loss was 11.66, close to the 11.5 that a model guessing uniformly over the vocabulary would score.

On the CPU, where the script compiles the model, five runs shared one checkpoint folder, with `--save-every 5 --keep-checkpoints 3` and dropout set to 0 so that weights could be compared. `torch.compile` was switched to its `eager` backend, which still returns the `OptimizedModule` wrapper; the code its default backend generates hangs on the machine these docs were checked on.

1. The first run was killed at step 17, in epoch 3.
2. The second resumed from the save made after 15 steps (2 epochs and 3 batches), with every weight tensor and the whole optimizer state equal to that file. It was interrupted after 22 steps and saved that exact position.
3. The third resumed from it (3 epochs and 4 batches), finished epoch 4, wrote `ep4`, deleted `ep1`, and was killed half-way through writing a rolling file.
4. The fourth reported that file as unreadable, resumed from the other one at 22 steps, and trained to the end.
5. The fifth found five completed epochs and trained nothing.

The final weights and `ep5` were equal in every tensor to those of a run of the same script that was never interrupted. `generate.py` loaded the final weights and, with those removed, the newer rolling file.

Also verified:

- With every checkpoint in the folder cut short, the script listed each file it could not read, stopped with `RuntimeError: None of the checkpoints ... could be read`, and left the folder as it was.
- A folder holding only `ep1` to `ep3` written before the rolling checkpoint existed was resumed from `ep3`.
- `tests/test_checkpoint.py` trains a small model three more times: interrupted inside the second epoch, stopped by an error inside the second epoch, and interrupted between the last batch of the first epoch and its checkpoint. After the resume all three end with weights bit-identical to an uninterrupted run. The tests fail when the sampler ignores the start position, when the trainer does not restore the batch count, when the first save may overwrite the file resumed from, and when unreadable checkpoints start a fresh run (each verified by making that change).

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
uv run python scripts/train_wikitext.py                                          # WikiText-103, five epochs
uv run python scripts/train_wikitext.py --data-dir data/fineweb-mix --epochs 1   # the FineWeb mix, one pass
uv run python scripts/train_wikitext.py --checkpoint-dir DIR                     # for example a folder on Google Drive
```

The script keeps its name, but it trains on whatever folder `--data-dir` names, as long as it holds a `train.bin` and a `validation.bin` in the format the prepare scripts write (see [02-data-pipeline.md](02-data-pipeline.md)).

| Option | Default | Effect |
|---|---|---|
| `--data-dir DIR` | `data/wikitext-103` | folder with `train.bin` and `validation.bin` |
| `--epochs N` | 5 | passes over the training data in total, counting those earlier runs completed |
| `--checkpoint-dir DIR` | `checkpoints` | where checkpoints and final weights are written, and where a resume looks |
| `--save-every N` | 2000 | training steps between saves of the rolling checkpoint |
| `--keep-checkpoints N` | 10 | end-of-epoch checkpoints to keep; older ones are deleted |

In order, it:

1. Reads its options.
2. Picks the device with `get_device()`. Seeds with 123, builds the model, moves it to the device, prints its size.
3. Takes the data folder from `--data-dir`. If that folder is on Google Drive, meaning its real path starts with `/content/drive/`, it copies the folder to `/content/data/<folder name>` once and reads from there. A folder that is already on the Colab machine's own disk is read where it is.
4. Builds three memory-mapped loaders and prints the batch counts of the first and the last: the training loader with `seed=123`, a shuffled loader over the same file for evaluation's training batches, and the validation loader.
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
| Training order | reshuffled every epoch from `seed=123` plus the epoch number |
| Optimizer | `AdamW(lr=5e-4, weight_decay=0.1)` |
| Epochs (`num_epochs`) | `--epochs`, 5 by default, in total across resumes |
| `eval_frq` / `eval_iter` | 500 / 10 |
| Sample prompt | "The secret to training large models is" |

Batches per epoch, verified by building the loaders on the real files:

| Data | Training tokens | Training batches per epoch | Validation batches |
|---|---|---|---|
| `data/fineweb-mix` | 1,000,075,641 | 244,159 | 256 |
| `data/wikitext-103` | 120,196,189 | 29,344 | 62 |

One pass over the FineWeb mix is therefore about 1.7 times the steps of five WikiText-103 epochs (146,720).

### Resume

```python
checkpoint, latest_checkpoint = load_latest_checkpoint(checkpoint_dir)
if checkpoint is not None:
    state_dict = checkpoint['model_state_dict']
    unwrapped_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(unwrapped_state_dict)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    resume = training_progress(checkpoint, latest_checkpoint)
    del checkpoint, state_dict, unwrapped_state_dict
```

`load_latest_checkpoint` (`src/mini_llm/training/checkpoint.py`) tries, in this order: the two rolling files, the one modified more recently first, and then the end-of-epoch checkpoints from the highest epoch number down. It returns the first that `torch.load` can read.

- **A file that cannot be read is reported and skipped.** The script prints `Could not read <file>: <error>` and tries the next one.
- **It never starts over on top of checkpoints.** If the folder holds checkpoints and none can be read, it raises. A run starts from scratch only when the folder holds none.
- **The rolling files come first.** Deleting end-of-epoch checkpoints does not move a run back; to go back to an epoch, remove the two rolling files and the later end-of-epoch checkpoints.
- **The file is loaded onto the CPU** and released once its contents have been copied into the model and the optimizer. An out-of-memory error on the GPU can therefore not be mistaken for a damaged file.
- **A run with a compiled model can be resumed.** `torch.compile` wraps the model in an `OptimizedModule` whose state-dict keys start with `_orig_mod.`, so a checkpoint saved from a compiled model carries that prefix. The code strips it and loads the result into the plain model, which is compiled only afterwards. Loading into the wrapper fails; that was [known issue 2](08-known-issues.md#2-resuming-into-a-compiled-model-failed-fixed).

The script then prints, for example, `Starting training from epoch 3, after 3 of its batches...`.

### Interrupt and final save

`Ctrl-C` during training is handled in the trainer, which saves the position it has reached to the rolling checkpoint; see [Checkpoints](#checkpoints). The script catches the interrupt after that and prints that the next run resumes from the rolling checkpoint. It no longer writes `gpt_wikitext_checkpoint_interrupted.pth`, and a file of that name left by an earlier version is ignored.

After the `try` block the script saves `model.state_dict()` to `gpt_wikitext_124M.pth` in the checkpoint folder. That includes the case where training was interrupted and the case where nothing was trained because all epochs were already done, so the file existing does not mean training finished; see [known issue 3](08-known-issues.md#3-the-interrupt-checkpoint-was-never-resumed-fixed-and-final-weights-are-saved-regardless). When the model was compiled, the keys in this file carry the `_orig_mod.` prefix; `generate.py` strips it on load.

### Files it reads and writes

| Path | Direction | When |
|---|---|---|
| `<data dir>/train.bin`, `validation.bin` | read | always |
| `/content/data/<folder name>/` | written | on Colab, first run, when the data folder is on Google Drive |
| `<checkpoint dir>/gpt_wikitext_checkpoint_latest_a.pth`, `..._latest_b.pth` | read and written | resume; every `--save-every` steps, at the end of each epoch, on `Ctrl-C` |
| `<checkpoint dir>/gpt_wikitext_checkpoint_ep{N}.pth` | read, written and deleted | resume when no rolling file can be read; end of each epoch; once more than `--keep-checkpoints` exist |
| `<checkpoint dir>/gpt_wikitext_124M.pth` | written | at exit |

## Colab notebook

`train_colab.ipynb` sits at the repo root and is meant to be opened in VS Code with the Colab extension, so the file stays local while its cells run on a Colab GPU. Nothing it produces lands on the local machine.

| Step | What the cell does |
|---|---|
| Settings and helpers | Defines the paths, the data settings `DATA_DIR`, `TRAIN_TOKENS` (1,000,000,000), `EDU_SHARE` (0.5) and `EPOCHS` (1), the checkpoint settings `SAVE_EVERY` (2000) and `KEEP_CHECKPOINTS` (10), and the helper functions. Safe to re-run at any time; it starts nothing. |
| 1. Check the GPU | Stops if the kernel has no CUDA device. |
| 2. Mount Google Drive | Mounts Drive at `/content/drive` and stops if the mount is missing, since the checkpoints would otherwise go to the temporary disk. |
| 3. Get the code | Clones the private repo into `/content/mini-llm` with a GitHub token typed at a prompt. The token is passed to git through the environment and is not stored. |
| 4. Install | `pip install -e` of the clone with the `data` extra. |
| 5. Prepare the data | Runs `prepare_fineweb.py` with `TRAIN_TOKENS` and `EDU_SHARE`. The script does nothing when the folder already holds data prepared with those settings. |
| 6. Start training | Starts `train_wikitext.py` with `--checkpoint-dir <Drive folder>`, `--data-dir`, `--epochs`, `--save-every` and `--keep-checkpoints` as a detached background process, with its output appended to a log file on Drive. Does nothing if training is already running. |
| 7. Watch progress | Parses the evaluation lines out of the log and plots train and validation loss; a second cell shows the end of the log and the checkpoint folder. |
| 8. Stop training | The first cell sends the interrupt that `Ctrl-C` would send, on which the trainer saves its position. The second, run once training has stopped, calls `drive.flush_and_unmount()`, which returns when everything written to Drive has been uploaded. |

Where things live during a Colab run:

| What | Where | Kept after the session |
|---|---|---|
| The clone, the installed package | `/content/mini-llm` | no |
| `train.bin`, `validation.bin`, `manifest.json` | `/content/mini-llm/data/fineweb-mix/` | no; rebuilt by step 5 in every session, from the same pinned version of the data |
| Checkpoints and final weights | `mini-llm/checkpoints/` on Google Drive | yes |
| The training log | `mini-llm/train.log` on Google Drive | yes |

To sample from a checkpoint locally, download it from Drive into `checkpoints/` and pass it to `generate.py --weights`; see [06-inference.md](06-inference.md#generatepy).

Verified locally: the clone cell against the real repository with a valid and an invalid token; the start, status and stop cells against a stand-in script that forks two workers; and the progress cell against a log written by the real trainer, where it parsed 138 of 138 evaluation lines. **Not run:** the notebook on Colab itself, so the GPU check, the Drive mount, the install, the data preparation and the flush cell are untested there.

### Across sessions

A new session resumes from the newer rolling checkpoint on Drive, at the step it was saved at, and continues the interrupted epoch in the same order.

- **What a session can lose.** When the machine is taken away, the steps since the last rolling save: at most `SAVE_EVERY`, 2,000 by default. One pass over the FineWeb mix is 244,159 steps. When training is stopped with the stop cell, nothing, provided the flush cell is run before disconnecting.
- **Space on Drive.** With the defaults a five-epoch run ends with five end-of-epoch checkpoints, two rolling files and the final weights, about 21 GB. `KEEP_CHECKPOINTS = 2` brings that to about 12.5 GB. A checkpoint the script deletes goes to Drive's Trash, where it keeps using space until the Trash is emptied (Colab's FAQ).
- **Saves are uploaded in the background.** A save that has not finished uploading when the machine disappears is lost, and the next session resumes from the other rolling file, one save earlier.

Not established, because it needs a run on Colab:

- **Replaced versions.** Whether Drive counts the replaced versions of an overwritten file against the quota. Its API documentation says such versions are purged, typically after 30 days, and says only of versions marked "Keep Forever" that they count. The storage figure in Drive after the first few saves will show it.
- **Save time.** How long a 2.88 GB save takes on the Drive mount, and with it how much of each `SAVE_EVERY` interval is spent saving.
- **Modification times.** Whether modification times keep their order after Drive is mounted in a new session. Resume picks the newer rolling file by modification time; if the order were wrong, the run would resume from the other file, one save earlier.
- **Steps per second on a Colab GPU.** The log gets a time-stamped line every 500 steps, so the first minutes of a run show it.
