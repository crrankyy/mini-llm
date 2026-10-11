# 02: Data pipeline

Token ids become training examples by sliding a fixed-size window over them. The input is the window; the target is the same window shifted one token to the right.

There are two dataset classes: `CustomDataset` holds everything in memory and is used for the small text file, and `MemMapDataset` reads a binary file on demand and is used for the prepared token files. Two scripts prepare such files: `prepare_fineweb.py` for the training data proper and `prepare_wikitext.py` for the much smaller WikiText-103.

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

## Preparing the FineWeb mix

`scripts/prepare_fineweb.py` builds the training data for the full-size model: by default 1 billion tokens, half from FineWeb-Edu and half from FineWeb. The work is done by `prepare_corpus` in `src/mini_llm/corpus.py`; the script supplies the two sources and writes the manifest. It needs the `data` extra.

```sh
uv run python scripts/prepare_fineweb.py                       # 1,000,000,000 tokens into data/fineweb-mix/
uv run python scripts/prepare_fineweb.py --tokens 100000000    # a smaller set
```

| Option | Default | Effect |
|---|---|---|
| `--tokens N` | 1,000,000,000 | training tokens to prepare; each takes 4 bytes on disk |
| `--validation-tokens N` | 1,000,000 | tokens held out for validation |
| `--edu-share X` | 0.5 | share taken from FineWeb-Edu; the rest comes from FineWeb |
| `--output-dir DIR` | `data/fineweb-mix` | where the three files are written |

### The two sources

| | FineWeb-Edu | FineWeb |
|---|---|---|
| Hub id | `HuggingFaceFW/fineweb-edu` | `HuggingFaceFW/fineweb` |
| Subset used | `sample-10BT`, 14 files | `sample-10BT`, 15 files |
| Contents | web pages that a classifier rated as educational | the general web: English pages that passed language, quality and duplicate filters |
| Licence tag on the Hub | ODC-By | ODC-By |
| Documents in the subset | 9,672,101 | 14,868,862 |
| Tokens per document in what was taken | 1,007 | 671 |
| Layout of a file | blocks of about 6,000 documents from one crawl; the part taken from a file holds 3 or 4 crawls, and other files hold other crawls | 7 or 8 crawls, finely interleaved in runs of at most a few hundred documents |

FineWeb is not the raw web. It is still English only, de-duplicated and quality-filtered; FineWeb-Edu then keeps the roughly 8% of those pages that score as educational. The mix is there so that the model sees dense, clean text and also the web as people actually write it.

Both sources are pinned to one revision in the script, so building the files again gives the same files. A training run that is resumed in a new Colab session depends on the data being the same as before.

### What the script does

1. **Opens the files of a source one at a time** as a stream through `datasets`, reading only the `id` and `text` columns. Only what is read is downloaded.
2. **Takes the same amount from every file**: `tokens x share / files`. In FineWeb-Edu the part taken from one file comes from only 3 or 4 crawls, and each file holds different ones, so drawing from all 14 covers many more crawls than the start of one file would. Within a file the documents are taken from its start.
3. **Holds out validation documents first.** The first documents of each file go to validation until that file's part of `--validation-tokens` is full. The ones after them go to training until its part of `--tokens` is full, and then the stream is closed. A validation document is therefore never trained on.
4. **Skips a document it already has.** FineWeb-Edu is a subset of FineWeb, so a page can be in both samples. Documents are compared by their `id`.
5. **Tokenizes with `cl100k_base`** through `encode_ordinary_batch`, which treats everything in a page as ordinary text, including the characters `<|endoftext|>`. It then appends the real end-of-text token, id 100,257, after every document.
6. **Appends each training document to `train.bin` as it goes**, so memory does not grow with the amount prepared.
7. **Shuffles the validation documents** with a fixed seed and writes `validation.bin`. Evaluation reads only the first 40,960 tokens of that file (see [08-known-issues.md](08-known-issues.md#smaller-items)), and without the shuffle those would all come from one file.
8. **Writes `manifest.json` last.** Its presence is what says the token files are complete; a new run deletes it before it starts writing.

| File | Contents |
|---|---|
| `train.bin`, `validation.bin` | flat `uint32` token ids, the same format `prepare_wikitext.py` writes, with id 100,257 after every document |
| `manifest.json` | the tokenizer, the token and document counts per source, the duplicates skipped, and under `request` the options and the two revisions |

Things that follow from that:

- **A second run with the same options does nothing.** It finds the manifest and compares its `request` entry. With different options it rebuilds.
- **A run that stops part-way leaves no manifest**, so the next one starts again.
- **The result is slightly larger than asked for.** Every file overshoots its part by less than one document.
- **The smallest file of a source limits how much that source can give**, because every file gives the same amount. If a file runs out, the script prepares less and says so. Not measured where that limit is; the last file of each source is about a quarter the size of the others.
- **Within a file only the beginning is used.** For FineWeb-Edu that reaches at least 30 different crawls between 2013 and 2024, but not every crawl in the subset.
- **The script ends with `os._exit`.** Closing a stream early leaves the Arrow library fetching the next part of that file in the background, and if the interpreter shuts down meanwhile the process never exits (verified: without it the script printed "Done!" and then hung, every time; [apache/arrow#45214](https://github.com/apache/arrow/issues/45214)). By then everything is on disk. An error is still printed, with exit code 1.

### Verified run

The script was run for real on 2026-10-11 on the Intel MacBook with its default options.

| | |
|---|---|
| Training tokens | 1,000,075,641, a 4.0 GB file |
| Training documents | 1,241,484: 496,403 from FineWeb-Edu and 745,081 from FineWeb |
| Tokens per source | 500,044,165 and 500,031,476 |
| Validation | 1,045,077 tokens in 1,352 documents, 485 and 867 |
| Duplicates skipped | 18, all of them FineWeb pages already taken from FineWeb-Edu |
| Time | 12.7 minutes |
| Peak memory | 784 MB |

Checked on the result:

- `train.bin` holds exactly 1,241,484 separator tokens, one per document, and ends with one.
- The crawl of every row in the part taken was read from three files of each source. FineWeb: 7 or 8 crawls per file, spread over 2013 to 2023, in runs of at most 289 rows. FineWeb-Edu: 3 or 4 crawls per file in blocks of about 6,000 rows, for example 2013, 2017, 2020 and 2023 in the first file and 2015, 2018 and 2021 in the eighth. A thinner sample across all 14 FineWeb-Edu files found 30 different crawls from 2013 to 2024.
- Documents decoded from several places in the file read as ordinary pages: a chemistry lesson and a health article in the FineWeb-Edu part, a careers post, a football report and a news item in the FineWeb part.
- The loaders `train_wikitext.py` builds give 244,158 training batches per epoch and 255 validation batches. The 40,960 tokens that evaluation reads hold 46 documents.
- `train_wikitext.py`, with the model shrunk as described in [05-training.md](05-training.md#verified-run), trained on these files, was killed, and resumed from its rolling checkpoint.
- Six separate builds of a 30-million-token set wrote byte-identical files.
- With the Hub unreachable the script printed the error and ended with exit code 1 in 24 seconds, leaving no manifest.

`tests/test_corpus.py` checks the properties above on small made-up sources, without the network. Not run: the script on Colab, where the time it takes is not known.

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
| `train_wikitext.py` | `create_memmap_dataloader` | `<--data-dir>/*.bin` | `--batch-size`, 4 by default | 1,024 | 1,024 | seeded order per epoch / yes | 2 |

Validation loaders in both scripts use `shuffle=False, drop_last=False`. `train_wikitext.py` builds a third loader over `train.bin` with `shuffle=True, drop_last=True`, from which evaluation draws its training batches.

`train_wikitext.py` cuts windows as long as the model's context. `train_tiny.py` keeps windows of 256 tokens, a quarter of the context, because its text is too short for longer ones: the validation part would not fill one window of 1,024.

For `train_tiny.py` the numbers are (verified): 20,480 characters, split at character 18,432; 4,423 training tokens giving 17 windows and 8 batches; 522 validation tokens giving 2 windows and 1 batch.
