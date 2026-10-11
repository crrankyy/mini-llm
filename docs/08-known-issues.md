# 08: Known issues

Places where the code does something different from what it appears to do. Each item gives the location, the behaviour, and how it was established.

Line numbers refer to the layout after the reorganization of 2026-10-11. Issues keep their numbers once assigned, so fixed ones stay in the list, marked as fixed. "Verified" means the behaviour was reproduced with torch 2.2.2; "from reading" means it was not run.

## Summary

| # | Issue | Location | Status |
|---|---|---|---|
| 1 | Causal mask was never applied | `src/mini_llm/model/attention.py:54` | fixed 2026-10-11 |
| 2 | Resuming into a compiled model failed | `scripts/train_wikitext.py:98`, `:109` | fixed 2026-10-11 |
| 3 | Interrupt checkpoint is never resumed; final weights saved regardless | `scripts/train_wikitext.py:89`, `:126`, `:139` | open; verified |
| 4 | `architecture.py` could not be imported | `study/p04_gpt/architecture.py:5` | fixed 2026-10-11 |
| 5 | Config name and vocabulary size do not match the model | `src/mini_llm/config.py:1-2` | open; verified |
| 6 | `train_tiny.py` left checkpoints that the other scripts picked up | `scripts/train_tiny.py:148` | fixed 2026-10-11 |
| 7 | `CustomDataset` can emit a short final target | `src/mini_llm/data.py:12` | open; verified |
| 8 | Unknown-token handling in the stage 1 tokenizers | `study/p01_tokenization/encoder_decoder.py:20`, `:35` | open; verified |
| 9 | `SelfAttention_v2` builds its dropout inside `forward` | `study/p03_attention/self_attention.py:53` | open; verified |
| 10 | Checkpoints are loaded with `weights_only=False` | `scripts/generate.py:42`, `scripts/train_wikitext.py:93` | open; from reading |
| 11 | Sampling an id the tokenizer cannot decode stopped the run | `src/mini_llm/generation.py:53` | fixed 2026-10-11 |
| 12 | `train_wikitext.py` cannot start its DataLoader workers on macOS | `scripts/train_wikitext.py:64`, `:74` | open; verified |

Before a long training run, issue 3 and the first two rows under [Smaller items](#smaller-items) are the ones that can cost work.

## Details

### 1. The causal mask was never applied (fixed)

Fixed on 2026-10-11. Until then the code was:

```python
mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
attention_scores.masked_fill(mask_bool, -torch.inf)
```

`masked_fill` returns a new tensor. The result was not assigned, so `attention_scores` reached softmax unmasked and every position attended to the positions after it. The second line now assigns the result back to `attention_scores`, both in the library and in the copy at `study/p03_attention/self_attention.py:126`. `tests/test_causal_mask.py` fails if the assignment is removed again (verified).

What followed from the bug:

- **Training and validation loss.** Position `t` was trained to predict token `t + 1` and could read token `t + 1` through attention. The loss could fall by copying, so neither the training nor the validation number measured next-token prediction.
- **Existing weights.** Any weights produced before the fix were trained without the mask and need retraining.

Check, with the package installed:

```python
import torch
from mini_llm.model.gpt import GPTModel

cfg = dict(vocab_size=50, context_length=8, embedding_dimension=16,
           n_heads=2, n_layers=2, drop_rate=0.0, qkv_bias=False)
torch.manual_seed(0)
model = GPTModel(cfg).eval()

a = torch.tensor([[1, 2, 3, 4, 5, 6]])
b = a.clone()
b[0, -1] = 9  # change only the last token

with torch.no_grad():
    print((model(a) - model(b)).abs().amax(dim=-1)[0])
```

Before the fix this printed `tensor([0.0890, 0.0892, 0.0777, 0.0741, 0.0840, 1.2428])`. It now prints `tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.2433])`: the first five values are exactly zero, because nothing before the last position can see the change.

### 2. Resuming into a compiled model failed (fixed)

Fixed on 2026-10-11. `train_wikitext.py` used to compile the model before it looked for a checkpoint. By the time the checkpoint was loaded, `model` was the `OptimizedModule` wrapper, whose keys all start with `_orig_mod.`, while the resume code strips that prefix from the checkpoint. `load_state_dict` raised `RuntimeError: Error(s) in loading state_dict for OptimizedModule: Missing key(s) in state_dict: "_orig_mod.token_embedding.weight", ...`, so a run with a compiled model could not be continued by a later run.

The script now loads the checkpoint into the plain model (line 98) and compiles afterwards (line 109). The prefix is still stripped, so checkpoints written by a compiled model load, including ones written before the fix.

Verified on torch 2.2.2 on CPU, with a shrunken model and a real `OptimizedModule`: a run was killed in the middle of epoch 3, and the next run resumed from `ep2` with every weight tensor and the whole optimizer state equal to the checkpoint. Before the fix that second run stopped with the error above. The runs used `torch.compile`'s `eager` backend, because the code its default backend generates hangs on that machine. Not run on Colab or on CUDA.

### 3. The interrupt checkpoint is never resumed, and final weights are saved regardless

- The `Ctrl-C` handler writes `gpt_wikitext_checkpoint_interrupted.pth` and prints "Next time you run this, it will resume from here." The resume pattern on line 89 is `gpt_wikitext_checkpoint_ep*.pth`, which does not match that name. The next run resumes from the last completed epoch, and the interrupted epoch starts over.
- Lines 135 to 140 run after the `try` block whenever the script has not died from an error. `gpt_wikitext_124M.pth` is written after an interrupt, and also when the script found all epochs already complete. The file existing does not mean training finished.
- `generate.py` looks for `gpt_wikitext_124M.pth` first. If it is absent, it takes the last of `sorted(glob("gpt_wikitext_checkpoint_*.pth"))`; `..._interrupted.pth` sorts after every `..._ep{N}.pth` (verified), so the interrupted checkpoint wins over the epoch checkpoints. Passing `--weights` avoids the search.

### 4. `study/p04_gpt/architecture.py` could not be imported (fixed)

Fixed on 2026-10-11. Line 5 imported from `sfeedforward`, which never existed, so the script failed with `ModuleNotFoundError`. The reorganization rewrote its imports to `mini_llm.model.layers` and `mini_llm.model.block`, and it now runs to the end (verified).

### 5. The config name and vocabulary size do not match the model

- `GPT_CONFIG_124M` builds a model with 239,365,632 parameters (verified). The output file is also named `gpt_wikitext_124M.pth`.
- `vocab_size` is 100,352 while `cl100k_base` has 100,277 tokens (verified). The 75 extra rows in the token embedding and in the output head are never selected by the tokenizer.
- The output head is a separate 77M-parameter matrix; it is not tied to the token embedding (verified).

This matters for memory, for checkpoint size, and for anyone comparing against GPT-2 small. See [04-model.md](04-model.md#parameter-budget). The extra rows are also part of issue 11.

### 6. `train_tiny.py` left checkpoints that the other scripts picked up (fixed)

Fixed on 2026-10-11. The trainer wrote `gpt_wikitext_checkpoint_ep{N}.pth` into the working directory for every caller, so the ten files `train_tiny.py` produced matched the resume pattern of `train_wikitext.py` and the fallback of `generate.py`.

The trainer now takes `checkpoint_dir`, and `train_tiny.py` passes `checkpoints/tiny`, a folder the other two scripts do not search.

### 7. `CustomDataset` can emit a short final target

```python
for i in range(0, len(token_ids) - max_length + 1, stride):
```

When `len(token_ids) - max_length` is a multiple of `stride`, the last window starts at `len(token_ids) - max_length`. Its input is full length but its target, shifted by one, runs off the end and is one token short.

Verified: 16 tokens with `max_length=8, stride=4` gives samples of lengths `(8, 8), (8, 8), (8, 7)`, and a `DataLoader` over them raises `RuntimeError: stack expects each tensor to be equal size`. With 17 tokens all samples are `(8, 8)`. The same loop is in `study/p02_data_loading/dataloader.py:12`.

The current `train_tiny.py` data does not trigger it. Whether a given text does depends on its exact token count.

### 8. Unknown-token handling in the stage 1 tokenizers

- `SimpleTokenizer.encoder` returns `None` for a token that is not in the vocabulary. The error only appears later, in `decoder`, as `ValueError: None is not in list`.
- `SimpleTokenizerV2.encoder` inserts the string `"<|unk|>"` into the id list, not the id of `<|unk|>`. Its output is a mix of integers and strings.

Verified with `"Hello, the"`: `[None, 5, 988]` and `['<|unk|>', 5, 988]`. These classes are used only by `test_simple_tokenizer.py`.

### 9. `SelfAttention_v2` builds its dropout inside `forward`

```python
dropout = nn.Dropout(p=0.1)
attention_weights = dropout(attention_weights)
```

A module created inside `forward` is not registered on the parent, so `model.eval()` does not reach it and it is always in training mode. The rate is fixed at 0.1.

Verified: in eval mode, two calls on the same input return different outputs, and the module's registered children are only `W_query`, `W_key` and `W_value`. The class exists only in the stage 3 study script, where `MultiHeadAttentionWrapper` and the demo use it; the model does not.

### 10. Checkpoints are loaded with `weights_only=False`

Both `torch.load` calls disable the restricted unpickler. That is what lets them read the checkpoint dict, and it also means a `.pth` file can run code when loaded. Load only files produced by these scripts.

### 11. Sampling an id the tokenizer cannot decode stopped the run (fixed)

Fixed on 2026-10-11. The output head has 100,352 entries, but `tokenizer.decode` raises `KeyError` for 91 of those ids: 100,256, 100,261 to 100,275, and the 75 from 100,277 to 100,351 (verified with tiktoken 0.14.0). Nothing stops the model from sampling one of them.

- **In the trainer.** `generate_and_print_sample` runs at the end of each epoch, before the checkpoint is saved (`src/mini_llm/training/trainer.py:64` and `:68`). A sample containing one of these ids ended the run, and that epoch's checkpoint was never written.
- **In `generate.py`.** The same exception ended the prompt loop.

Both now decode through `decode_token_ids` (`src/mini_llm/generation.py:53`). It decodes one id at a time and writes an id that has no token as its number, for example `<|100289|>`. When every id has a token it returns exactly what `tokenizer.decode` returns. Sampling is unchanged: the model can still pick these ids, and the marker shows when it does.

Verified with a model made to prefer two of these ids. Before the fix `train_wikitext.py` died at the end of the first epoch with `KeyError: 'Invalid token for decoding: 100289'` and left no checkpoint, and `generate.py` died the same way. Now the trainer prints the sample with the markers in it and saves all five checkpoints, and `generate.py` prints its response. `tests/test_generation.py` covers the decoding and the trainer's sample, and fails if the old decode is put back (verified).

The trainer still prints the sample before it saves the checkpoint, so any other error in the sample would cost that epoch's checkpoint.

These ids never occur as training targets, so training pushes their logits down and the markers should become rarer as the model learns. How often they appear after a full epoch on WikiText-103 was not measured.

### 12. `train_wikitext.py` cannot start its DataLoader workers on macOS

The script builds its loaders with `num_workers=2` and has no `if __name__ == "__main__":` guard. macOS starts worker processes by spawning a new interpreter, which re-runs the script from the top inside each worker.

Verified with a shrunken model: each worker printed the device banner and "Starting training" again, then raised `RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase`, and the main process stopped with `DataLoader worker exited unexpectedly`.

Linux starts workers by forking, which does not re-run the script, so Colab is not affected. That was not run.

## Smaller items

| Item | Location | Notes |
|---|---|---|
| A checkpoint is written only at the end of an epoch | `src/mini_llm/training/trainer.py:68` | work since the last finished epoch is lost when a run stops; see also issue 3 |
| A checkpoint that was cut off stops the next run | `scripts/train_wikitext.py:93` | `torch.save` writes straight to the final name and the resume loads the newest file with no fallback. Verified by truncating the newest checkpoint: the next run stopped with `RuntimeError: PytorchStreamReader failed reading zip archive`, and resumed from the checkpoint before it once the damaged file was deleted |
| Loaded weights stay in memory after a resume | `scripts/train_wikitext.py:93` | `checkpoint` is never released, so a resumed run keeps a second copy of the weights on the device, 957 MB at full size (from reading). The optimizer state is shared with the optimizer, not copied (verified on torch 2.2.2) |
| Step counters restart on resume | `src/mini_llm/training/trainer.py:14` | `global_step` and `tokens_seen` are saved in each checkpoint and never read back (verified) |
| Whole split tokenized in memory | `scripts/prepare_wikitext.py:19-24` | all ids go into a Python list, then an array, before the `memmap` write |
| No document separators | `scripts/prepare_wikitext.py:20-22` | articles are concatenated; no end-of-text token between them |
| Last window dropped | `src/mini_llm/data.py:37` | `MemMapDataset` can omit one valid final window (verified) |
| Context size read from the global config | `scripts/generate.py:56` | `generate_text` ignores the size of the model it is given |
| Optimizer state loaded for inference | `scripts/generate.py:42` | given a full checkpoint, `map_location=device` moves the AdamW state to the device before it is dropped |
| Unused imports and variables | the three scripts that build a model | `train_tiny.py`: `nn`, `batch`. `train_wikitext.py`: `nn`, `textgen_v1`, `text_to_token_ids`, `token_ids_to_text`, `calc_loss_loader`. `generate.py`: `sys` |
| Two test files | `tests/` | the causal mask and the decoding of sampled ids; training, resume and data loading have no tests |

## Duplicated code

The study scripts keep their own copies of code that also lives in the library. When changing one copy, check whether the other should change too.

| What | Copies |
|---|---|
| `tokenize` | `study/p01_tokenization/simple_tokenizer.py`, `study/p01_tokenization/encoder_decoder.py` |
| `CustomDataset`, `create_dataloader` | `study/p02_data_loading/dataloader.py`, `src/mini_llm/data.py` (the second adds `drop_last` and `num_workers`) |
| `MultiHeadedAttention` | `study/p03_attention/self_attention.py`, `src/mini_llm/model/attention.py` |
| The GPT network | `DummyGPT` in `study/p04_gpt/architecture.py`, `GPTModel` in `src/mini_llm/model/gpt.py` |
| Top-k and temperature sampling loop | `textgen_v2` in `src/mini_llm/generation.py`, `generate_text` in `scripts/generate.py` |
| `GPT_CONFIG_124M` | `src/mini_llm/config.py`, `study/p04_gpt/architecture.py` (different values) |
| Parameter count and size printout | `scripts/train_tiny.py`, `scripts/train_wikitext.py` |

The reorganization removed three duplicates: the copies of the loss functions and of `textgen_v1` under stage 4, and the second copy of the three earlier attention classes.
