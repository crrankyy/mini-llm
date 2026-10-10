# 08: Known issues

Places where the code does something different from what it appears to do, ordered by how much they affect results. Each item gives the location, the behaviour, and how it was established.

Line numbers refer to commit `0bb727c`. Nothing listed here has been changed in the code. "Verified" means the behaviour was reproduced on CPU with torch 2.2.2; "from reading" means it was not run.

## Summary

| # | Issue | Location | Status |
|---|---|---|---|
| 1 | Causal mask is never applied | `src/p03_attention/attention_modules.py:125` | verified |
| 2 | Resuming into a compiled model fails | `train_wikitext.py:73`, `:95` | verified on torch 2.2.2 |
| 3 | Interrupt checkpoint is never resumed; final weights saved regardless | `train_wikitext.py:86`, `:116`, `:123` | sort order verified, rest from reading |
| 4 | `architecture.py` cannot be imported | `src/p04_gpt/architecture.py:5` | verified |
| 5 | Config name and vocabulary size do not match the model | `config.py:1-2` | verified |
| 6 | `test.py` leaves checkpoints that the other scripts pick up | `src/utils/trainer.py:63` | sort order verified, rest from reading |
| 7 | `CustomDataset` can emit a short final target | `src/utils/dataloader.py:12` | verified |
| 8 | Unknown-token handling in the stage 1 tokenizers | `src/p01_tokenization/encoder_decoder.py:20`, `:35` | verified |
| 9 | `SelfAttention_v2` builds its dropout inside `forward` | `src/p03_attention/attention_modules.py:52` | verified |
| 10 | Checkpoints are loaded with `weights_only=False` | `test_inference.py:14`, `train_wikitext.py:90` | from reading |

## Details

### 1. The causal mask is never applied

```python
mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
attention_scores.masked_fill(mask_bool, -torch.inf)
```

`masked_fill` returns a new tensor. The result is not assigned, so `attention_scores` reaches softmax unmasked and every position attends to the positions after it. The same line is in the copy at `src/p03_attention/self_attention.py:126`; the model imports the one in `attention_modules.py`.

What follows from it:

- **Training and validation loss.** Position `t` is trained to predict token `t + 1` and can read token `t + 1` through attention. The loss can fall by copying, so neither the training nor the validation number measures next-token prediction.
- **Generation.** The last position has no later tokens to read. The model is sampled under conditions it was not trained for.
- **Existing weights.** Any weights produced by this code were trained without the mask. How much that degrades their output was not measured.

Reproduce from the repo root:

```python
import torch
from src.model import GPTModel

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

Observed: `tensor([0.0890, 0.0892, 0.0777, 0.0741, 0.0840, 1.2428])`. With a working mask the first five values are exactly zero, because nothing before the last position can see the change.

The usual form assigns the result: `attention_scores = attention_scores.masked_fill(mask_bool, -torch.inf)`. Weights trained before such a change learned to rely on later tokens, so they would need retraining.

### 2. Resuming into a compiled model fails

`train_wikitext.py` compiles the model on line 73 and loads the checkpoint on line 95. At that point `model` is the `OptimizedModule` wrapper, whose keys all start with `_orig_mod.`. The code strips that prefix from the checkpoint and then loads the stripped keys into the wrapper.

Verified on torch 2.2.2: `load_state_dict` raises `RuntimeError: Error(s) in loading state_dict for OptimizedModule: Missing key(s) in state_dict: "_orig_mod.token_embedding.weight", ...`. The same stripped dict loads into an uncompiled model.

So resume works only when `torch.compile` raised and the script fell back to eager mode. This was not checked on the torch version Colab installs.

### 3. The interrupt checkpoint is never resumed, and final weights are saved regardless

- The `Ctrl-C` handler writes `gpt_wikitext_checkpoint_interrupted.pth` and prints "Next time you run this, it will resume from here." The resume pattern on line 86 is `gpt_wikitext_checkpoint_ep*.pth`, which does not match that name. The next run resumes from the last completed epoch, and the interrupted epoch starts over.
- Lines 120 to 124 run after the `try` block in every case. `gpt_wikitext_124M.pth` is written after an interrupt, and also when the script found all epochs already complete. The file existing does not mean training finished.
- `test_inference.py` looks for `gpt_wikitext_124M.pth` first. If it is absent, it takes the last of `sorted(glob("gpt_wikitext_checkpoint_*.pth"))`; `..._interrupted.pth` sorts after every `..._ep{N}.pth` (verified), so the interrupted checkpoint wins over the epoch checkpoints.

### 4. `src/p04_gpt/architecture.py` cannot be imported

Line 5 imports from `sfeedforward`, which does not exist. Every other file imports `LayerNorm`, `GELU` and `FeedForward` from `src.p04_gpt.feedforward`. Verified: the script fails with `ModuleNotFoundError` whether run by path or with `-m`.

### 5. The config name and vocabulary size do not match the model

- `GPT_CONFIG_124M` builds a model with 239,365,632 parameters (verified). The output file is also named `gpt_wikitext_124M.pth`.
- `vocab_size` is 100,352 while `cl100k_base` has 100,277 tokens (verified). The 75 extra rows in the token embedding and in the output head are never selected by the tokenizer.
- The output head is a separate 77M-parameter matrix; it is not tied to the token embedding (verified).

None of this breaks anything. It matters for memory, for checkpoint size, and for anyone comparing against GPT-2 small. See [04-model.md](04-model.md#parameter-budget).

### 6. `test.py` leaves checkpoints that the other scripts pick up

The checkpoint name `gpt_wikitext_checkpoint_ep{N}.pth` is hardcoded in the trainer, so `test.py`, which trains on `basic-data.txt`, writes `ep1` through `ep10` into the working directory.

If `train_wikitext.py` is run in the same directory afterwards (from reading, apart from the sort order):

1. Its glob matches those files. String sorting puts `ep9` last, not `ep10` (verified).
2. It loads the `ep9` weights and sets `start_epoch = 9`. If the model was compiled, it fails here instead, as in issue 2.
3. Since 9 is not less than `num_epochs = 5`, it prints "Already completed 5 epochs. Training is done!", skips training, and saves the `basic-data.txt` weights as `gpt_wikitext_124M.pth`.

`test_inference.py` would load the same `ep9` file through its fallback.

### 7. `CustomDataset` can emit a short final target

```python
for i in range(0, len(token_ids) - max_length + 1, stride):
```

When `len(token_ids) - max_length` is a multiple of `stride`, the last window starts at `len(token_ids) - max_length`. Its input is full length but its target, shifted by one, runs off the end and is one token short.

Verified: 16 tokens with `max_length=8, stride=4` gives samples of lengths `(8, 8), (8, 8), (8, 7)`, and a `DataLoader` over them raises `RuntimeError: stack expects each tensor to be equal size`. With 17 tokens all samples are `(8, 8)`. The same loop is in `src/p02_data_loading/dataloader.py:12`.

The current `test.py` data does not trigger it. Whether a given text does depends on its exact token count.

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

Verified: in eval mode, two calls on the same input return different outputs, and the module's registered children are only `W_query`, `W_key` and `W_value`. The class is used only by `MultiHeadAttentionWrapper` and the stage 3 demo, not by the model.

### 10. Checkpoints are loaded with `weights_only=False`

Both `torch.load` calls disable the restricted unpickler. That is what lets them read the checkpoint dict, and it also means a `.pth` file can run code when loaded. Load only files produced by these scripts.

## Smaller items

| Item | Location | Notes |
|---|---|---|
| Step counters restart on resume | `src/utils/trainer.py:12` | `global_step` and `tokens_seen` are saved in each checkpoint and never read back (verified) |
| Whole split tokenized in memory | `prepare_wikitext.py:19-24` | all ids go into a Python list, then an array, before the `memmap` write |
| No document separators | `prepare_wikitext.py:20-22` | articles are concatenated; no end-of-text token between them |
| Last window dropped | `src/utils/dataloader.py:37` | `MemMapDataset` can omit one valid final window (verified) |
| Context size read from the global config | `test_inference.py:28` | `generate_text` ignores the size of the model it is given |
| `device.py` is unused | `device.py` | no MPS branch; no script imports it |
| Unused imports and variables | `test.py`, `train_wikitext.py` | `test.py`: `nn`, `batch`. `train_wikitext.py`: `nn`, `textgen_v1`, `text_to_token_ids`, `token_ids_to_text`, `calc_loss_loader` |
| Outputs not ignored by git | `.gitignore` | `*.pth` and `data/wikitext-103/` are not listed |
| No dependency manifest or tests | repo root | see [07-running.md](07-running.md#environment) |

## Duplicated code

Later stages copy earlier code forward. When changing one copy, check whether the other should change too.

| What | Copies |
|---|---|
| `tokenize` | `src/p01_tokenization/simple_tokenizer.py`, `src/p01_tokenization/encoder_decoder.py` |
| `CustomDataset`, `create_dataloader` | `src/p02_data_loading/dataloader.py`, `src/utils/dataloader.py` (the second adds `drop_last` and `num_workers`) |
| The four attention classes | `src/p03_attention/self_attention.py`, `src/p03_attention/attention_modules.py` |
| `calc_loss_batch`, `calc_loss_loader` | `src/p04_gpt/loss.py`, `src/utils/loss.py` |
| `textgen_v1` | `src/p04_gpt/textgen.py`, `src/utils/text2token.py` |
| The GPT network | `DummyGPT` in `src/p04_gpt/architecture.py`, `GPTModel` in `src/model.py` |
| Top-k and temperature sampling loop | `textgen_v2` in `src/utils/text2token.py`, `generate_text` in `test_inference.py` |
| `GPT_CONFIG_124M` | `config.py`, `src/p04_gpt/architecture.py` (different values) |
| Parameter count and size printout | `test.py`, `train_wikitext.py` |

Issue 1 exists in both copies of the attention classes. Only the copy in `attention_modules.py` affects the model.
