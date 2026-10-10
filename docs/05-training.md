# 05: Training

Training is next-token prediction: the model's logits at each position are compared with the token that follows it, using cross-entropy. The shared pieces are in `src/utils/`; `test.py` and `train_wikitext.py` are the two scripts that call them.

Read [08-known-issues.md](08-known-issues.md#1-the-causal-mask-is-never-applied) alongside this page. Because the attention mask is not applied, the losses these functions report do not measure next-token prediction.

## Loss (`src/utils/loss.py`)

### `calc_loss_batch(input_batch, target_batch, model, device)`

Moves both tensors to `device`, runs the model, and returns the mean cross-entropy over all `B x T` positions:

```python
logits = model(input_batch)
loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
```

`logits.flatten(0, 1)` is `(B*T, V)` and `target_batch.flatten()` is `(B*T,)`. The returned loss is a tensor that still carries its graph, so the caller can call `backward()` on it.

### `calc_loss_loader(data_loader, model, device, num_batches=None)`

Averages `calc_loss_batch` over the first `num_batches` batches of a loader, or over all of them if `num_batches` is `None`. It returns `nan` for an empty loader and a Python float otherwise. It does not switch the model to eval mode or disable gradients; the caller does that.

`src/p04_gpt/loss.py` contains the same two functions and is not imported anywhere.

## Evaluation (`src/utils/evaluator.py`)

`evaluate_model(model, train_loader, val_loader, device, eval_iter)` puts the model in eval mode, computes `calc_loss_loader` on `eval_iter` batches from each loader under `torch.no_grad()`, puts the model back in train mode, and returns `(train_loss, val_loss)`.

The training loader is shuffled, so each call scores a different random set of training batches.

## The training loop (`src/utils/trainer.py`)

```python
train_model_v1(model, train_loader, val_loader, optimizer, device, num_epochs,
               eval_frq, eval_iter, start_context, tokenizer, start_epoch=0)
```

Returns `(train_losses, val_losses, track_tokens_seen)`, three lists with one entry per evaluation.

Per batch:

| Device | Steps |
|---|---|
| CUDA | `zero_grad`; forward and loss inside `torch.autocast(device_type='cuda', dtype=torch.float16)`; `scaler.scale(loss).backward()`; `scaler.unscale_(optimizer)`; clip gradient norm to 1.0; `scaler.step(optimizer)`; `scaler.update()` |
| CPU | `zero_grad`; forward and loss; `loss.backward()`; clip gradient norm to 1.0; `optimizer.step()` |

The CUDA path uses mixed precision with a `torch.cuda.amp.GradScaler`. Gradients are unscaled before clipping so that the 1.0 threshold applies to their true size. The CUDA path was not run for these docs.

After each batch, `tokens_seen` grows by `input_batch.numel()` and `global_step` by one. `global_step` starts at -1, so the first batch is step 0.

**Every `eval_frq` steps**, starting with step 0, it calls `evaluate_model`, appends to the three lists, and prints:

```text
Ep 1 (Step 000005) [2026-10-10 23:06:17]: Train loss 11.657, Val loss 11.665
```

**At the end of every epoch** it prints a 50-token sample from `start_context` (see [06-inference.md](06-inference.md#generate_and_print_samplemodel-tokenizer-device-start_context)) and saves a checkpoint.

There is no learning-rate schedule and no gradient accumulation. The learning rate is whatever the optimizer was built with.

### Checkpoints

Written to the current working directory as `gpt_wikitext_checkpoint_ep{N}.pth`, where `N` is the 1-based epoch number. The name is hardcoded in the trainer, so `test.py` writes files with this name too.

| Key | Value |
|---|---|
| `epoch` | number of completed epochs |
| `model_state_dict` | `model.state_dict()` |
| `optimizer_state_dict` | `optimizer.state_dict()` |
| `global_step` | last step index of this call |
| `tokens_seen` | tokens processed in this call |

`global_step` and `tokens_seen` are saved but never read back. Each call to `train_model_v1` starts both from scratch, so after a resume the step counter in the log restarts at 0 (verified).

**Size.** In a small-model check, a checkpoint was 3.02 times the size of the float32 weights, because AdamW keeps two extra tensors per parameter. For the full model, whose weights are 957 MB (decimal), that extrapolates to about 2.9 GB per checkpoint. This figure was not measured at full size.

### Verified run

With a one-layer, 16-dimensional model, the real tokenizer and `basic-data.txt`, on CPU: two epochs produced evaluations at steps 0, 5, 10 and 15, `tokens_seen` values of 512, 3,072, 5,632 and 8,192, and files `..._ep1.pth` and `..._ep2.pth`. A second call with `start_epoch=2, num_epochs=1` wrote `..._ep3.pth`. The starting loss was 11.69, close to `ln(100352) = 11.52`, which is what a model guessing uniformly would score.

## `test.py`

A module-level script, hardcoded to CPU. It was not run for these docs because it trains the full 239M-parameter model. In order, it:

1. Builds the tokenizer and stacks two four-token sentences into a tensor named `batch`, which is not used afterwards.
2. Seeds with 123, builds `GPTModel(GPT_CONFIG_124M)`, and prints the parameter count and size.
3. Generates 10 tokens greedily from "Every effort moves you" with the untrained model.
4. Reads `data/basic-data.txt`, prints its character and token counts, and splits it 90/10 **by characters**.
5. Builds the two loaders (see [02-data-pipeline.md](02-data-pipeline.md#which-loader-each-script-uses)), prints every batch shape, and prints the initial train and validation loss.
6. Builds a **second, new** `GPTModel` and an `AdamW` optimizer, and trains it.

| Setting | Value |
|---|---|
| Optimizer | `AdamW(lr=0.0004, weight_decay=0.1)` |
| Epochs | 10 |
| `eval_frq` / `eval_iter` | 5 / 5 |
| Batches per epoch | 8 |
| Sample prompt | "Every effort moves you" |

It leaves ten `gpt_wikitext_checkpoint_ep*.pth` files in the working directory, roughly 29 GB by the estimate above. Those files match the resume pattern of `train_wikitext.py`; see [known issue 6](08-known-issues.md#6-testpy-leaves-checkpoints-that-the-other-scripts-pick-up).

## `train_wikitext.py`

A module-level script. It was not run as a whole for these docs. In order, it:

1. Picks `cuda` if available, otherwise `cpu`. Seeds with 123, builds the model, moves it to the device, prints its size.
2. Sets `data_dir = "data/wikitext-103"`. If `/content` exists, which is how it detects Colab, it copies that folder to `/content/data/wikitext-103` once and reads from there, so the data sits on the local disk and not on mounted Drive.
3. Builds the memory-mapped loaders and prints their batch counts.
4. Calls `torch.compile(model)` inside a `try`; if that raises, it prints a message and continues uncompiled.
5. Builds `AdamW(lr=5e-4, weight_decay=0.1)`.
6. Looks for a checkpoint to resume from.
7. Trains the remaining epochs, or prints that training is already complete.
8. Saves the final weights.

| Setting | Value |
|---|---|
| Batch size | 16 |
| Window / stride | 256 / 256 |
| `num_workers` | 2 |
| Optimizer | `AdamW(lr=5e-4, weight_decay=0.1)` |
| Epochs (`num_epochs`) | 5 in total, across resumes |
| `eval_frq` / `eval_iter` | 500 / 10 |
| Sample prompt | "The secret to training large models is" |

The number of batches per epoch depends on the size of `train.bin`, which was not generated here.

### Resume

```python
checkpoints = sorted(glob.glob("gpt_wikitext_checkpoint_ep*.pth"))
if checkpoints:
    latest_checkpoint = checkpoints[-1]
    checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    unwrapped_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(unwrapped_state_dict)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
```

`torch.compile` wraps the model in an `OptimizedModule` whose state-dict keys start with `_orig_mod.`. The code strips that prefix before loading.

- **Granularity is one epoch.** Training restarts at the beginning of the first unfinished epoch.
- **The latest file is chosen by string sort.** `ep10` sorts before `ep2`, so this picks the wrong file once there are ten or more. With `num_epochs = 5` that does not happen from this script alone.
- **Loading into a compiled model fails.** By the time the checkpoint is loaded, `model` is already the compiled wrapper, which expects the prefix that was just stripped. Verified on torch 2.2.2: `load_state_dict` raises `RuntimeError` listing every `_orig_mod.*` key as missing. The same stripped dict loads into an uncompiled model. Resume therefore works only when `torch.compile` fell back to eager mode. Not checked on the torch version Colab installs.

### Interrupt and final save

`Ctrl-C` during training is caught. The handler saves `gpt_wikitext_checkpoint_interrupted.pth` with `epoch` set to the epoch the run started from, and prints that the next run will resume from it. It will not: the resume pattern is `..._ep*.pth`, which this name does not match.

After the `try` block, in every case, the script saves `model.state_dict()` to `gpt_wikitext_124M.pth`. That includes the case where training was interrupted and the case where nothing was trained because all epochs were already done. When the model was compiled, the keys in this file carry the `_orig_mod.` prefix; `test_inference.py` strips it on load.

### Files it reads and writes

| Path | Direction | When |
|---|---|---|
| `data/wikitext-103/train.bin`, `validation.bin` | read | always |
| `/content/data/wikitext-103/` | written | on Colab, first run |
| `gpt_wikitext_checkpoint_ep{N}.pth` | read and written | resume; end of each epoch |
| `gpt_wikitext_checkpoint_interrupted.pth` | written | on `Ctrl-C` |
| `gpt_wikitext_124M.pth` | written | at exit |
