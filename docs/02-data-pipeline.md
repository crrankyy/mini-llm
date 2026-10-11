# 02: Data pipeline

Token ids become training examples by sliding a fixed-size window over them. The input is the window; the target is the same window shifted one token to the right.

There are two dataset classes: `CustomDataset` holds everything in memory and is used for the small text file, and `MemMapDataset` reads a binary file on demand and is used for WikiText-103.

## Stage 2 study scripts (`study/p02_data_loading/`)

### `dataloader.py`

The first version of `CustomDataset` and `create_dataloader`. The class is identical to the one in `src/mini_llm/data.py`; the factory has fewer parameters (`txt, batch_size=4, max_length=256, stride=128, shuffle=True`).

`main()` prints the first two batches for `max_length=8, stride=4`, once with `batch_size=1` and once with `batch_size=4`, so the overlap between consecutive windows is visible.

### `embedding.py`

Creates `nn.Embedding(8, 4)` with seed 123, prints its weight matrix, and looks up ids `[1, 2, 3, 4]`. It shows that an embedding layer is a row lookup.

### `embed_dataloader.py`

Takes one batch from the dataloader (`batch_size=8`, `max_length=128`, `stride=64`) and passes it through `nn.Embedding(tokenizer.n_vocab, 256)`, giving a `(8, 128, 256)` tensor. There is no positional embedding in this demo; that first appears in `GPTModel`.

It imports `study.p02_data_loading.dataloader`, so it has to be run as a module from the repo root (verified; see [07-running.md](07-running.md)).

## `src/mini_llm/data.py`

This is the version the entry points use.

### `CustomDataset(txt, tokenizer, max_length, stride)`

Tokenizes the whole text in `__init__` and stores every window as a pair of tensors:

```python
for i in range(0, len(token_ids) - max_length + 1, stride):
    input_chunk = token_ids[i:i + max_length]
    target_chunk = token_ids[i + 1:i + max_length + 1]
```

- `stride == max_length` gives non-overlapping windows; a smaller stride gives overlap.
- Tokens after the last full window are dropped.
- **Edge case (verified):** when `len(token_ids) - max_length` is an exact multiple of `stride`, the last window's target is one token short. With 16 tokens, `max_length=8`, `stride=4`, the third sample is `(8, 7)`, and batching it fails with `RuntimeError: stack expects each tensor to be equal size`. With 17 tokens all three samples are `(8, 8)`. `train_tiny.py` does not hit this with the current data.

### `create_dataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0)`

Builds a `cl100k_base` tokenizer, a `CustomDataset`, and a `torch.utils.data.DataLoader`. Each batch is `(input_ids, target_ids)`, both `(B, max_length)` and `int64`.

### `MemMapDataset(bin_file_path, max_length, stride)`

Opens the file with `np.memmap(..., dtype=np.uint32, mode='r')`, so nothing is loaded until a sample is requested.

```python
self.num_samples = (len(self.data) - self.max_length) // self.stride

def __getitem__(self, idx):
    start_idx = idx * self.stride
    chunk = self.data[start_idx : start_idx + self.max_length + 1].astype(np.int64)
    x = torch.tensor(chunk[:-1], dtype=torch.long)
    y = torch.tensor(chunk[1:], dtype=torch.long)
    return x, y
```

Each sample reads `max_length + 1` tokens and splits them into input and target. Verified with a 1,000-token file and `max_length = stride = 256`: 2 samples, sample 0 has `x = 0..255` and `y = 1..256`, both `int64`.

The sample count is conservative. In that check a third full window (tokens 512 to 768) would have fit, and 487 trailing tokens were never read. On a corpus-sized file this costs at most one window.

### `ResumableSampler(num_samples, seed)`

A sampler whose order can be rebuilt by a later run. `__iter__` seeds a generator with `seed + epoch`, draws one permutation of all the samples, and yields it from index `start` on. `set_position(epoch, start)` sets both; a new sampler is at epoch 0, start 0.

Verified: two samplers with the same seed and epoch give the same order, another epoch gives a different one, and after `set_position(epoch=3, start=40)` the sampler yields exactly the tail of epoch 3's order from its 41st sample. A loader with two worker processes follows the same order as one without.

The trainer uses it to continue an epoch that an earlier run did not finish; see [05-training.md](05-training.md#checkpoints).

### `create_memmap_dataloader(bin_file_path, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0, seed=None)`

The same wrapper for `MemMapDataset`. No tokenizer is involved, since the file already holds token ids.

With `seed` set, the loader is built with a `ResumableSampler` and `shuffle` is ignored. With `seed=None` it shuffles, or not, as `shuffle` says.

## Preparing WikiText-103

`scripts/prepare_wikitext.py` has one function, `prepare_data()`, run under a `main()` guard. It needs the `data` extra (`datasets` and `tqdm`). It was run twice for these docs: once with `load_dataset` replaced by a few lines of text, and once for real on 2026-10-11. The real run wrote 120,196,189 training tokens (481 MB) and 252,975 validation tokens (1 MB), took 115 seconds on the Intel MacBook, and peaked at 6.1 GB of memory, because every id goes into a Python list first.

1. `load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")` downloads the dataset through Hugging Face `datasets`. A comment in the file records why the `Salesforce/` namespace is used: the plain `wikitext` name was deprecated.
2. Creates `data/wikitext-103/`.
3. For the `train` and `validation` splits (the `test` split is not used), it walks every row, skips rows that are blank after `strip()`, encodes the rest with `tokenizer.encode(text, allowed_special="all")`, and extends one Python list with the ids.
4. Converts the list to a `uint32` NumPy array and writes it through `np.memmap(filename, mode='w+')` to `data/wikitext-103/{split}.bin`.

Things that follow from that code:

- **File format.** A `.bin` file is a flat array of native-endian `uint32` token ids with no header: 4 bytes per token (verified: 1,000 tokens is 4,000 bytes). The reader must know the dtype, and `MemMapDataset` hardcodes the same one.
- **No document boundaries.** Rows are concatenated as they come; no `<|endoftext|>` is inserted between articles.
- **Memory.** A whole split is tokenized into an in-memory Python list, then copied into an array, before anything is written. `memmap` is used for the write and for later reads, so it bounds memory during training, not during preparation.
- **Ignored by git.** `data/wikitext-103/` is listed in `.gitignore`.

Verified with the stand-in text: `train.bin` held exactly the ids the tokenizer produces for the non-blank rows, and `MemMapDataset` read it back with each target equal to its input shifted by one.

## Which loader each script uses

| Script | Loader | Source | `batch_size` | `max_length` | `stride` | Train `shuffle` / `drop_last` | `num_workers` |
|---|---|---|---|---|---|---|---|
| `train_tiny.py` | `create_dataloader` | `data/basic-data.txt`, split 90/10 by characters | 2 | 256 | 256 | yes / yes | 0 |
| `train_wikitext.py` | `create_memmap_dataloader` | `data/wikitext-103/*.bin` | 16 | 256 | 256 | seeded order per epoch / yes | 2 |

Validation loaders in both scripts use `shuffle=False, drop_last=False`. `train_wikitext.py` builds a third loader over `train.bin` with `shuffle=True, drop_last=True`, from which evaluation draws its training batches.

For `train_tiny.py` the numbers are (verified): 20,480 characters, split at character 18,432; 4,423 training tokens giving 17 windows and 8 batches; 522 validation tokens giving 2 windows and 1 batch.
