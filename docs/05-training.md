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
               resume=None, eval_train_loader=None, accumulate=1, lr_schedule=None)
```

| Parameter | Effect |
|---|---|
| `save_frq` | Steps between saves of the rolling checkpoint. `None`, the default, switches the rolling checkpoint off; `train_tiny.py` leaves it off. |
| `keep_checkpoints` | How many end-of-epoch checkpoints to keep. Older ones are deleted each time a new one is written. `None` keeps them all. |
| `resume` | The position to continue from, as `training_progress` returns it for a loaded checkpoint. It restores the step and token counters, the number of batches already done in the current epoch, the number of weight updates made, and the gradient scaler. The caller loads the weights and the optimizer state itself. |
| `eval_train_loader` | The loader evaluation takes its training batches from. Defaults to `train_loader`. |
| `accumulate` | Batches whose gradients are added up for one weight update. 1, the default, updates after every batch. |
| `lr_schedule` | A function from the number of weight updates already made to the learning rate of the next one. `None`, the default, leaves the optimizer's rate alone. |

Returns `(train_losses, val_losses, track_tokens_seen)`, three lists with one entry per evaluation.

It creates `checkpoint_dir` before the first batch, so a path that cannot be created fails at the start and not after an epoch.

The work is split between what happens for every batch and what happens once per weight update, which is made after every `accumulate` batches:

| Device | Per batch | Per update |
|---|---|---|
| CUDA | forward and loss inside `torch.autocast(device_type='cuda', dtype=torch.float16)`; `scaler.scale(loss / group_size).backward()` | set the learning rate; `scaler.unscale_(optimizer)`; clip gradient norm to 1.0; `scaler.step(optimizer)`; `scaler.update()`; `zero_grad` |
| CPU and MPS | forward and loss; `(loss / group_size).backward()` | set the learning rate; clip gradient norm to 1.0; `optimizer.step()`; `zero_grad` |

The CUDA path uses mixed precision with a `torch.cuda.amp.GradScaler`. Gradients are unscaled before clipping so that the 1.0 threshold applies to their true size. The CUDA path was not run for these docs; it ran on Colab on 2026-10-11, see [The Colab run of 2026-10-11](#the-colab-run-of-2026-10-11). MPS takes the second path in full precision; torch 2.2.2 has no autocast for it.

Four counters describe the position: `tokens_seen`, `global_step` (the index of the last finished batch; it starts at -1, so the first batch is step 0), `batches_done` (batches finished in the current epoch) and `updates_done` (weight updates made). They move together, once per update and by the whole group of batches, in one statement. A "step" is still one batch. After a resume all four continue from the checkpoint.

**Evaluation is due every `eval_frq` steps**, starting with step 0, and happens at the first update at or after the step it is due at. It calls `evaluate_model`, appends to the three lists, and prints the losses and the learning rate of the last update:

```text
Ep 1 (Step 000001) [2026-10-11 20:23:35]: Train loss 11.518, Val loss 11.517, lr 2.00e-04
```

**A save of the rolling checkpoint is due every `save_frq` steps** and happens at the first update at or after that step; see [Checkpoints](#checkpoints). With one batch per update, both happen at exactly the step they are due at.

**At the end of every epoch** it prints a 50-token sample from `start_context` (see [06-inference.md](06-inference.md#generate_and_print_samplemodel-tokenizer-device-start_context)), saves the end-of-epoch checkpoint, deletes the end-of-epoch checkpoints beyond `keep_checkpoints`, and saves the rolling checkpoint again. The order matters: if printing the sample fails, that epoch's checkpoint is not written, and a later run continues from the last rolling save. The one known way for it to fail, a sampled id the tokenizer cannot decode, is fixed; see [known issue 11](08-known-issues.md#11-sampling-an-id-the-tokenizer-cannot-decode-stopped-the-run-fixed).

### Gradient accumulation

With `accumulate` above 1 the trainer adds up the gradients of that many batches before it changes the weights. Each batch's loss is divided by the number of batches in its group, so the sum is the gradient of the mean loss over all of them. The update is the one a single batch `accumulate` times as large would make, without that batch having to fit in memory. The computation per token is the same as before; the weights move less often, on a steadier gradient.

- **The last update of an epoch takes the batches that are left** and divides by their number.
- **Only whole updates are saved.** The counters move when an update has been made, so no saved position includes a batch whose gradient exists only in memory. A run that stops between two updates repeats the batches since the last one, at most `accumulate - 1`.
- **Evaluation and saves wait for the next update**, as described above.

Verified by `tests/test_checkpoint.py`: 24 samples trained in batches of 1 with `accumulate=5` end with the weights that batches of 5 give, to within 1e-6, including the last update from the 4 samples left over. The test uses plain gradient descent, because Adam rescales gradients and would hide one of the wrong size.

### Learning-rate schedule (`src/mini_llm/training/schedule.py`)

`learning_rate_at(update, total_updates, peak, warmup_updates, final)` returns the rate for the next weight update, given how many have been made:

| Updates made | Rate |
|---|---|
| fewer than `warmup_updates` | `peak * (update + 1) / warmup_updates`: it rises in equal steps, the first update is not made at zero, and the last warmup update is made at `peak` |
| up to `total_updates` | falls from `peak` to `final` along half a cosine wave |
| `total_updates` or more | stays at `final` |

The trainer calls `lr_schedule(updates_done)` once before every update and writes the result into every parameter group of the optimizer. The rate depends only on `updates_done`, which is in every checkpoint, so a resumed run continues on the same curve and no scheduler state has to be saved.

`train_wikitext.py` builds the schedule from its options; see [Updates and learning rate](#updates-and-learning-rate). `train_tiny.py` passes neither a schedule nor `accumulate`, and trains at a constant rate with one update per batch.

### Optimizer (`src/mini_llm/training/optimizer.py`)

`create_optimizer(model, lr, weight_decay)` returns `AdamW` with two parameter groups and betas of 0.9 and 0.95:

| Group | Parameters | In the full model | Weight decay |
|---|---|---|---|
| tensors with two or more dimensions | every weight matrix, the token and position embeddings | 74 tensors, 162,791,424 parameters | `weight_decay` |
| tensors with one dimension | biases, the scale and shift of every `LayerNorm` | 86 tensors, 93,696 parameters | 0 |

- **Decay on the matrices only.** Weight decay shrinks a parameter a little at every update. Applied to a normalisation scale, it pulls the scale towards 0, when the value that leaves the layer's input unchanged is 1. The tied matrix appears once in `model.parameters()` and is decayed once.
- **A second beta of 0.95.** AdamW divides each update by a running average of the squared gradient. With PyTorch's default of 0.999 that average covers about the last 1,000 updates, with 0.95 about the last 20, so it follows a change in the size of the gradients sooner.
- **Both come from the published run the learning rate is taken from**; see [Updates and learning rate](#updates-and-learning-rate). Until 2026-10-12 the script decayed every parameter and used the default betas. Their effect on this model has not been measured.

Verified by `tests/test_optimizer.py`: after one step with every gradient at zero, each matrix is smaller by the factor `1 - lr * weight_decay` and each vector is unchanged; every parameter is in exactly one group. `train_tiny.py` does not use this function.

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
| `updates_done` | weight updates made so far |
| `batch_size` | windows per batch in the run that wrote the file: the unit `batches_done` is counted in |

`training_progress` fills in defaults for keys a file does not have: a position at the start of an epoch, and one update per batch. No older file can be resumed in practice, though. The model's state dict changed on 2026-10-11 (see [03-attention.md](03-attention.md#what-carries-over-into-checkpoints)), and given a folder of checkpoints written before that, the script stops with `Unexpected key(s) in state_dict: "transformer_blocks.0.attention.mask"` (verified). The context length changed on 2026-10-12, and a checkpoint written between the two dates stops it with `size mismatch for position_embedding.weight` (verified with the rolling checkpoint of the Colab run). Both happen before it writes anything.

**The rolling checkpoint is two files written in turn.** Each save overwrites the older file in place, so a save that is cut off damages one file and leaves the other. The first save of a run never goes to the file the run resumed from, which until then is the only good copy. Nothing is renamed or deleted: on Google Drive a deleted file goes to the Trash and keeps counting against the storage quota, and whether a rename over an existing file does the same could not be established.

**On `Ctrl-C`** the trainer saves the position of the last weight update to the rolling checkpoint and raises the interrupt again. Batches processed since that update are not counted, because their gradients go with the process; the next run processes them again. If a save was in progress, it is redone into the same file. An interrupt can also arrive in the middle of an optimizer update; the state saved then holds whatever part of that update had been applied, while the position says it has not been made, so the next run makes it again.

**Pruning.** After each end-of-epoch checkpoint, the ones beyond the newest `keep_checkpoints` are deleted with `os.remove`. Only files named `gpt_wikitext_checkpoint_ep{N}.pth` in `checkpoint_dir` are touched, and they are ordered by `N` as a number.

**Continuing inside an epoch.** `train_wikitext.py` builds its training loader with `seed=123`, which gives the loader a `ResumableSampler` (see [02-data-pipeline.md](02-data-pipeline.md#resumablesamplernum_samples-seed)). The order of an epoch then depends only on the seed and the epoch number. The trainer starts each epoch with `sampler.set_position(epoch, batches_done * batch_size)`, so a resumed run visits exactly the windows the stopped run had not reached. That count is in batches, so every checkpoint records the batch size it was written with, and the trainer raises `ValueError` when the loader it is given has another one: at 100,000 batches of 4, a run resumed with batches of 8 would start at window 800,000 and skip 41% of the epoch. It raises before anything is written. A checkpoint that records no batch size is accepted. With an ordinary loader a position inside an epoch cannot be restored, and the trainer raises `ValueError`. Evaluation takes its training batches from `eval_train_loader`, an ordinary shuffled loader, so they remain a different random set at every evaluation.

**Size.** A checkpoint of the full model is 1.95 GB (measured), three times the 652 MB of float32 weights, because AdamW keeps two extra tensors per parameter. On the Intel MacBook's internal disk one save of the full model took 4.3 seconds and loading it back 2.4 seconds (each measured once). A run holds two rolling files, up to `keep_checkpoints` end-of-epoch files, and the final weights.

### Verified run

`train_wikitext.py` was run with the model shrunk to one 8-dimensional layer and a 32-token context, small stand-in token files, and its DataLoader workers switched off. The runs below were made on 2026-10-12, after the batch size became an option and the optimizer got its two parameter groups.

On MPS, where the model is not compiled, with the default options. At 4 windows per batch the stand-in file gives 24 batches per epoch, so every epoch was two updates, one from 16 batches and one from the 8 left:

- A first run trained five epochs and wrote `..._ep1.pth` to `..._ep5.pth`, the two rolling files and the final weights.
- A second run found the newer rolling file, printed "Already completed 5 epochs", and trained nothing.

The first loss it printed, after one update, was 11.515; a model guessing uniformly over the vocabulary would score 11.516.

On the CPU, where the script compiles the model, five runs shared one checkpoint folder, with `--save-every 4 --keep-checkpoints 3 --batch-size 16 --accumulate 2 --warmup-share 0.2`: 6 batches per epoch and 15 weight updates in all, at a rate that rises over the first three and falls after that. `torch.compile` was switched to its `eager` backend, which still returns the `OptimizedModule` wrapper; the code its default backend generates hangs on the machine these docs were checked on.

1. The first run was killed at step 17, in epoch 3.
2. The second resumed from the save made after 16 steps (2 epochs and 4 batches), with every weight tensor and the whole optimizer state equal to that file. It was interrupted after 23 batches, one batch into an update, and saved the position of the last update: 22 steps.
3. A start with `--batch-size 8` in between was refused: `ValueError: The checkpoint was written with a batch size of 16, and this run uses 8.` It left every file in the folder as it was.
4. The third resumed from the same save (3 epochs and 4 batches) and was killed half-way through writing its first rolling save.
5. The fourth reported that file as unreadable, resumed from the other one at 22 steps, and trained to the end, writing `ep4` and `ep5` and deleting `ep1` and `ep2`.
6. The fifth found five completed epochs and trained nothing.

The final weights and `ep5` were equal in every tensor to those of a run of the same script, with the same options, that was never interrupted. `generate.py` loaded the final weights and, with those removed, the newer rolling file.

Also verified:

- With every checkpoint in the folder cut short, the script listed each file it could not read, stopped with `RuntimeError: None of the checkpoints ... could be read`, and left the folder as it was. This was run before the model and the trainer changed on 2026-10-11; the code involved did not change, and `tests/test_checkpoint.py` covers the function.
- A folder of checkpoints written by the model as it was before 2026-10-11 is refused, as described under [Checkpoints](#checkpoints). So is one written with the 256-token context: the script, run as written at full size on the real data, printed 244,158 training batches and 15,260 weight updates, found the Colab run's rolling checkpoint, stopped with the size mismatch, and left the file unchanged.
- `tests/test_checkpoint.py` trains a small model through five stops: interrupted, and stopped by an error, inside the second epoch, each with one and with two batches per update; and interrupted between the last batch of the first epoch and its checkpoint. After the resume all five end with weights bit-identical to an uninterrupted run, at a learning rate that is different for every update. It also resumes a stopped run with half the batch size and expects the refusal with no file changed, and resumes from checkpoints whose batch size was removed.
- Those tests fail when the sampler ignores the start position, when the trainer does not restore the batch count, when the first save may overwrite the file resumed from, and when unreadable checkpoints start a fresh run (each verified by making that change).
- They also fail when the update count is not restored, when a saved position counts batches that have had no update, when gradients are not cleared after an update, when a batch's loss is not divided by the size of its group or is divided by `accumulate` in a short last group, and when the schedule's rate is not written into the optimizer (each verified the same way).
- Since 2026-10-12 they fail as well when a resume with another batch size is accepted, when the batch size is not written to or not read from the checkpoint, when a checkpoint without one is refused, when the scheduled rate reaches only the first parameter group, and when the schedule is asked once per group. `tests/test_optimizer.py` fails when the vectors are decayed, when the matrices are not, when the matrices are left out of the optimizer, and with PyTorch's default betas (each verified the same way).

## `train_tiny.py`

`scripts/train_tiny.py` is a module-level script that trains the full 163M-parameter model. On 2026-10-12 it was run as written on MPS, with only the checkpoint writes switched off: the ten epochs took just under 3 minutes, the training loss went from 10.94 to 6.05 and the validation loss from 10.86 to 7.05. In order, it:

1. Picks the device with `get_device()`.
2. Builds the tokenizer and stacks two four-token sentences into a tensor named `batch`, which is not used afterwards.
3. Seeds with 123, builds `GPTModel(GPT_CONFIG_124M)`, and prints the parameter count and size.
4. Generates 10 tokens greedily from "Every effort moves you" with the untrained model.
5. Reads `data/basic-data.txt`, prints its character and token counts, and splits it 90/10 **by characters**.
6. Builds the two loaders (see [02-data-pipeline.md](02-data-pipeline.md#which-loader-each-script-uses)) from windows of 256 tokens, prints every batch shape, and prints the initial train and validation loss. The windows are a quarter of the model's context because the text is too short for longer ones: its validation part holds 522 tokens.
7. Builds a **second, new** `GPTModel` and an `AdamW` optimizer, and trains it.

| Setting | Value |
|---|---|
| Optimizer | `AdamW(lr=0.0004, weight_decay=0.1)` on every parameter with PyTorch's default betas, at a constant rate, one update per batch |
| Epochs | 10 |
| `eval_frq` / `eval_iter` | 5 / 5 |
| Batches per epoch | 8 |
| Sample prompt | "Every effort moves you" |
| Checkpoint folder | `checkpoints/tiny` |

It leaves ten `gpt_wikitext_checkpoint_ep*.pth` files in `checkpoints/tiny`, about 19.5 GB at 1.95 GB each. Because that is a separate folder, `train_wikitext.py` and `generate.py` no longer pick these files up.

## `train_wikitext.py`

`scripts/train_wikitext.py` is a module-level script. On the machine these docs were checked on it was run end to end only as described under [Verified run](#verified-run): not at full size, and with `torch.compile` only through its `eager` backend. The version of 2026-10-11 ran at full size on a Colab GPU; see [The Colab run of 2026-10-11](#the-colab-run-of-2026-10-11). On macOS it cannot run as written; see [known issue 12](08-known-issues.md#12-train_wikitextpy-cannot-start-its-dataloader-workers-on-macos).

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
| `--batch-size N` | 4 | windows in one batch; each is as long as the model's context, 1,024 tokens |
| `--accumulate N` | 16 | batches whose gradients are added up for one weight update |
| `--lr X` | 6e-4 | peak learning rate |
| `--warmup-share X` | 0.03 | share of all weight updates during which the rate rises to its peak |

In order, it:

1. Reads its options.
2. Picks the device with `get_device()`. Seeds with 123, builds the model, moves it to the device, prints its size.
3. Takes the data folder from `--data-dir`. If that folder is on Google Drive, meaning its real path starts with `/content/drive/`, it copies the folder to `/content/data/<folder name>` once and reads from there. A folder that is already on the Colab machine's own disk is read where it is.
4. Builds three memory-mapped loaders and prints the batch counts of the first and the last: the training loader with `seed=123`, a shuffled loader over the same file for evaluation's training batches, and the validation loader.
5. Builds the optimizer with `create_optimizer(model, lr=<--lr>, weight_decay=0.1)` and the learning-rate schedule, and prints how many weight updates the run will make.
6. Looks in the checkpoint folder for a checkpoint to resume from, and loads it.
7. Unless the device is MPS, calls `torch.compile(model)` inside a `try`; if that raises, it prints a message and continues uncompiled.
8. Trains the remaining epochs, or prints that training is already complete.
9. Saves the final weights into the checkpoint folder.

Compilation is skipped on MPS because there `torch.compile` returns without error and the first forward pass then fails with `Device mps not supported` (verified on torch 2.2.2), outside the `try`. It comes after the resume because the compiled wrapper cannot load the checkpoint; see [Resume](#resume).

| Setting | Value |
|---|---|
| Batch size | `--batch-size`, 4 by default: 4,096 tokens |
| Window / stride | 1,024 / 1,024, the model's context |
| `num_workers` | 2 |
| Training order | reshuffled every epoch from `seed=123` plus the epoch number |
| Optimizer | `AdamW` with betas 0.9 and 0.95 and `weight_decay=0.1` on the matrices only, at the scheduled learning rate; see [Optimizer](#optimizer-srcmini_llmtrainingoptimizerpy) |
| Batches per update | `--accumulate`, 16 by default: 65,536 tokens |
| Learning rate | rises to `--lr` (6e-4) over the first `--warmup-share` (3%) of all updates, then falls along a cosine to a tenth of `--lr` |
| Epochs (`num_epochs`) | `--epochs`, 5 by default, in total across resumes |
| `eval_frq` / `eval_iter` | 500 / 10: every 500 steps, on 10 batches, which at the default batch size is 40 windows or 40,960 tokens |
| Sample prompt | "The secret to training large models is" |

Batches per epoch at the default batch size. The FineWeb row is verified by building the loaders on the real files; the WikiText row is worked out from its token counts, because those files were not on the machine when the batch changed from 16 x 256 to 4 x 1,024:

| Data | Training tokens | Training batches per epoch | Weight updates per epoch, at 16 batches each | Validation batches |
|---|---|---|---|---|
| `data/fineweb-mix` | 1,000,075,641 | 244,158 | 15,260 | 255 |
| `data/wikitext-103` | 120,196,189 | 29,344 | 1,834 | 62 |

One pass over the FineWeb mix is therefore about 1.7 times the steps of five WikiText-103 epochs (146,720).

### Updates and learning rate

```python
updates_per_epoch = math.ceil(len(train_dataloader) / args.accumulate)
total_updates = num_epochs * updates_per_epoch
warmup_updates = round(args.warmup_share * total_updates)

def lr_schedule(update):
    return learning_rate_at(update, total_updates, args.lr, warmup_updates, 0.1 * args.lr)
```

With the defaults, one pass over the FineWeb mix is 15,260 updates of 65,536 tokens, the last one from the 14 batches left over. The rate rises to 6e-4 over the first 458 and falls to 6e-5 by the last. The script prints the two counts when it starts.

- **The curve is worked out again every time the script starts**, from `--epochs`, `--batch-size`, `--accumulate`, `--lr` and `--warmup-share`. A run resumed with a different value continues on a different curve from the same update count, except for `--batch-size`, where the resume is refused (see [Checkpoints](#checkpoints)). Raising `--epochs` after a run has finished moves the rate back up before it falls again.
- **These defaults are choices, not measurements.** 16 batches per update, the peak rate and the warmup ran for 16,511 steps with a 256-token context on 2026-10-11, and the loss fell smoothly through the peak of the rate; see [The Colab run of 2026-10-11](#the-colab-run-of-2026-10-11). No run has been made with the 1,024-token context, the betas of 0.9 and 0.95 or decay on the matrices only.
- **Why 4 windows per batch.** 4 x 1,024 is the 4,096 tokens that 16 x 256 was, so a step, the save interval and an update cover as much text as before, and the step counts stay comparable with the earlier run. An update now averages over 64 windows where it had 256, so its gradient is somewhat noisier; this has not been measured.
- **Why 16 batches.** Accumulating gradients makes nothing faster on one GPU. Kaplan et al., *Scaling Laws for Neural Language Models* (2020), find that batches well below a critical size reach a given loss with the fewest tokens. Their fit puts that size at about 94,000 tokens at a loss of 5 and 449,000 at 3.6, measured with a different tokenizer.
- **Why 6e-4.** A published run of this transformer size on FineWeb-Edu used that peak with updates of 524,288 tokens in windows of 1,024, decay on the weight matrices only, and AdamW betas of 0.9 and 0.95. Since 2026-10-12 this script has the same windows, decay and betas; its updates are 65,536 tokens.

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
- **Another batch size is refused.** The trainer raises `ValueError` when the checkpoint was written with a different `--batch-size`; see [Checkpoints](#checkpoints). The script has printed its "Starting training" line by then, and stops without writing anything.
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
| Settings and helpers | Defines the paths, the data settings `DATA_DIR`, `TRAIN_TOKENS` (1,000,000,000), `EDU_SHARE` (0.5) and `EPOCHS` (1), the checkpoint settings `SAVE_EVERY` (2000) and `KEEP_CHECKPOINTS` (10), the training settings `BATCH_SIZE` (4), `ACCUMULATE` (16), `LEARNING_RATE` (6e-4) and `WARMUP_SHARE` (0.03), and the helper functions. Safe to re-run at any time; it starts nothing. |
| 1. Check the GPU | Stops if the kernel has no CUDA device. |
| 2. Mount Google Drive | Mounts Drive at `/content/drive` and stops if the mount is missing, since the checkpoints would otherwise go to the temporary disk. |
| 3. Get the code | Clones the repo into `/content/mini-llm`. The repository is public, so the cell asks for no GitHub token. If git is asked for a login, the cell stops with git's error. |
| 4. Install | `pip install -e` of the clone with the `data` extra. |
| 5. Prepare the data | Runs `prepare_fineweb.py` with `TRAIN_TOKENS` and `EDU_SHARE`, writing to the data folder on Drive. The script does nothing when the folder already holds data prepared with those settings. |
| 6. Start training | Starts `train_wikitext.py` with `--checkpoint-dir <Drive folder>`, `--data-dir`, `--epochs`, `--save-every`, `--keep-checkpoints`, `--batch-size`, `--accumulate`, `--lr` and `--warmup-share` as a detached background process, with its output appended to a log file on Drive. Does nothing if training is already running. |
| 7. Watch progress | Parses the evaluation lines out of the log and plots train and validation loss; a second cell shows the end of the log and the checkpoint folder. |
| 8. Stop training | The first cell sends the interrupt that `Ctrl-C` would send, on which the trainer saves its position. The second, run once training has stopped, calls `drive.flush_and_unmount()`, which returns when everything written to Drive has been uploaded. |

Where things live during a Colab run:

| What | Where | Kept after the session |
|---|---|---|
| The clone, the installed package | `/content/mini-llm` | no |
| `train.bin`, `validation.bin`, `manifest.json` | `mini-llm/data/fineweb-mix/` on Google Drive | yes |
| The copy of those files that training reads | `/content/data/fineweb-mix/` | no; copied from Drive again when training first starts in a session |
| Checkpoints and final weights | `mini-llm/checkpoints-1024/` on Google Drive | yes |
| The training log | `mini-llm/train-1024.log` on Google Drive | yes |

The folder and the log are named after the context length. The run of 2026-10-11, with a 256-token context, wrote to `mini-llm/checkpoints/` and `mini-llm/train.log`; the script cannot resume from those files, and the progress cell would plot both runs as one if they shared a log.

To sample from a checkpoint locally, download it from Drive into `checkpoints/` and pass it to `generate.py --weights`; see [06-inference.md](06-inference.md#generatepy).

Verified locally on 2026-10-12: the settings and clone cells against the real repository; the start, status and stop cells against a stand-in script that forks two workers, with the start cell passing `--batch-size 4`; and the progress cell against a log written by the real trainer. On 2026-10-11 the progress cell parsed 138 of 138 evaluation lines. The repository was private then and the clone cell passed a token to git; it has been made public since, and the cell no longer asks for one. Run on 2026-10-12 with no git configuration and no stored credentials, it cloned the repository without sending any, and given a repository name that GitHub wants a login for, it stopped within a second with `terminal prompts disabled` and left no folder behind.

On Colab, every cell of the notebook as it was on 2026-10-11 has run once; see [The Colab run of 2026-10-11](#the-colab-run-of-2026-10-11). The notebook with `BATCH_SIZE`, the new folder names and the clone that asks for no token has not been run there.

### Across sessions

A new session resumes from the newer rolling checkpoint on Drive, at the step it was saved at, and continues the interrupted epoch in the same order.

- **What a session can lose.** When the machine is taken away, the steps since the last rolling save: `SAVE_EVERY`, 2,000 by default, plus the few a save waits for its weight update. One pass over the FineWeb mix is 244,158 steps. When training is stopped with the stop cell, the batches since the last weight update, at most 15, which the next session processes again. That assumes the flush cell is run before disconnecting.
- **The training settings have to stay the same.** `EPOCHS`, `BATCH_SIZE`, `ACCUMULATE`, `LEARNING_RATE` and `WARMUP_SHARE` define the learning-rate curve; see [Updates and learning rate](#updates-and-learning-rate). A start with a different `BATCH_SIZE` is refused.
- **Space on Drive.** A checkpoint is 1.95 GB and the final weights 0.65 GB. A one-epoch run ends with one end-of-epoch checkpoint, two rolling files and the final weights, about 6.5 GB. A five-epoch run ends with about 14.3 GB, or about 8.5 GB with `KEEP_CHECKPOINTS = 2`. The training data takes another 4 GB. A checkpoint the script deletes goes to Drive's Trash, where it keeps using space until the Trash is emptied (Colab's FAQ). The folder of an earlier run counts as well: two rolling files and the weights of the 256-token run are 4.55 GB, which with the data and a finished one-epoch run comes to about 15 GB.
- **Saves are uploaded in the background.** A save that has not finished uploading when the machine disappears is lost, and the next session resumes from the other rolling file, one save earlier.

Not established, because it needs a run on Colab:

- **Replaced versions.** Whether Drive counts the replaced versions of an overwritten file against the quota. Its API documentation says such versions are purged, typically after 30 days, and says only of versions marked "Keep Forever" that they count. The storage figure in Drive after the first few saves will show it.
- **Save time.** How long a 1.95 GB save takes on the Drive mount, and with it how much of each `SAVE_EVERY` interval is spent saving.
- **Modification times.** Whether modification times keep their order after Drive is mounted in a new session. Resume picks the newer rolling file by modification time; if the order were wrong, the run would resume from the other file, one save earlier.
- **Memory and speed at 4 x 1,024.** A batch holds the 4,096 tokens it held at 16 x 256, which fitted the T4, but each window is four times as long. From a reading of PyTorch 2.11's source, not from a run: a T4 computes the attention in a kernel that does not build the 1,024 x 1,024 score matrix, in which case a step needs about the memory it needed before. If the log shows `CUDA out of memory`, set `BATCH_SIZE = 2`, `ACCUMULATE = 32` and `SAVE_EVERY = 4000`: an update and a save then cover as much text as before. Evaluation is counted in steps, so it would then come twice as often and read 20,480 tokens. For the speed, the log gets a time-stamped line every 500 steps; at 16 x 256 two of them were about 3 minutes apart.
- **The data copy.** How long the 4 GB copy from Drive to the Colab disk takes at the start of a session. Nothing compares the size of `train.bin` with the manifest, so an upload or a copy that was cut short would not be noticed.

### The Colab run of 2026-10-11

The first run at full size, made by the repository's owner with the notebook as it was that day: a 256-token context (162,295,296 parameters), batches of 16 x 256, 16 batches per update, a peak rate of 6e-4, a 3% warmup, weight decay on every parameter and PyTorch's default betas. It was stopped with the stop cell shortly after step 16,511 of 244,159, because the context length was about to change. What is known about it comes from the outputs saved in the notebook, from the owner's reading of the log, and from two files downloaded from Drive.

| | |
|---|---|
| Machine | Tesla T4, torch 2.11.0+cu130 |
| Cells | every cell ran, from the GPU check to the flush; the data step found the uploaded data and prepared nothing |
| Speed | about 3 minutes between two evaluation lines, which are 500 steps apart: about 11,400 tokens a second, or about 24 hours for one pass |
| Loss | 11.6 at the first evaluation; 4.730 on training text and 4.869 on validation text at step 16,511, the 34th evaluation; the curve was smooth through the peak of the learning rate at step 7,328 |
| Rolling checkpoint of step 16,000 | 1,947,761,429 bytes; loaded on the Intel MacBook; keys with the `_orig_mod.` prefix, so the model had been compiled with `torch.compile`'s default backend |
| Learning rate in that file | 0.0005982220889050127, exactly `learning_rate_at` for the 1,000th update |
| Gradient scaler in that file | scale 65,536 and 1,000 updates since the last change: none of the 1,000 updates was skipped for an overflow |
| Stop cell | the weights file written after the interrupt differs from the step-16,000 checkpoint in all 161 tensors, so the script reached its final save; the position saved on the interrupt was not looked at |
| Text sampled from both files | fluent word by word, no facts right; 13.5 tokens a second on the MacBook's GPU |

From the fifth evaluation on, validation sat 0.1 to 0.15 above training, with less than 7% of the training text seen once, so the model cannot have memorised it. Scored by token frequencies alone, the 40,960 validation tokens that evaluation reads come out at 7.89 against 7.82 for random training windows: the fixed sample is harder than average text, which accounts for part of the gap. The rest was not measured.

Not seen in that run: a resume in a second session, the time a save takes on the Drive mount, and the GPU's memory use.
