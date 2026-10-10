# 01: Tokenization

Stage 1 lives in `study/p01_tokenization/`. It builds a word-level tokenizer by hand, then switches to a byte-pair-encoding tokenizer from `tiktoken`. Only the `tiktoken` tokenizer is used by the rest of the repository.

## The regex tokenizer

Both `simple_tokenizer.py` and `encoder_decoder.py` define the same `tokenize` function:

```python
def tokenize(text):
    tokens = re.split(r'([,.;:!?_"()\']|--|\s)', text)
    tokens = [token for token in tokens if token.strip()]
    return tokens
```

It splits on punctuation, double dashes and whitespace, keeps the punctuation as separate tokens, and drops whitespace. On `data/basic-data.txt` it produces 4,690 tokens (verified).

`vocabulary_builder(tokens)` sorts the unique tokens and numbers them from 0. The two files differ here:

| File | Vocabulary |
|---|---|
| `simple_tokenizer.py` | unique tokens only |
| `encoder_decoder.py` | unique tokens, then `<\|unk\|>` and `<\|endoftext\|>` appended at the end |

For `basic-data.txt` the second version gives 1,132 entries, with `<|unk|>` at id 1130 and `<|endoftext|>` at id 1131 (verified).

## `simple_tokenizer.py`: functions

| Function | Returns | Unknown token |
|---|---|---|
| `encoder(text, vocab)` | `(ids, vocab)`, a tuple | raises `KeyError` |
| `decoder(encoded_text, vocab)` | a list of token strings, not a joined string | raises `KeyError` |

The rest of the file is a demo that runs at import: it reads `data/basic-data.txt`, prints the first 100 tokens and ids, and asserts that decoding the encoded text gives back the token list.

## `encoder_decoder.py`: classes

Both classes take a prebuilt `vocab` dict and expose `encoder(text)` and `decoder(encoded_text)`.

### `SimpleTokenizer`

- `encoder` uses `self.vocab.get(token)`, so an unknown token becomes `None` in the output list. It does not raise.
- `decoder` finds each id by scanning `list(self.vocab.values()).index(i)`, which rebuilds two lists per token. It returns a list of token strings. A `None` in the input raises `ValueError`.

Verified: `encoder("Hello, the")` returns `[None, 5, 988]`, and decoding that raises `ValueError: None is not in list`.

### `SimpleTokenizerV2`

- Builds `inverse_vocab` once in `__init__`, so decoding is a dict lookup.
- `encoder` is meant to map unknown tokens to `<|unk|>`, but it inserts the **string** `"<|unk|>"` into the id list, not the id 1130.
- `decoder` maps anything missing from `inverse_vocab` to `"<|unk|>"`, joins the tokens with spaces, then removes the space before punctuation with `re.sub(r'\s([,.;:!?_"()\']|--)', r'\1', text)`. It returns a string.

Verified: `encoder("Hello, the")` returns `['<|unk|>', 5, 988]` and `decoder` of that returns `'<|unk|>, the'`. The round trip works because the decoder's fallback happens to turn the stray string back into the same text; the encoded list is not all integers.

## `test_simple_tokenizer.py`

A script, not a test. It imports `encoder_decoder` by bare module name, which works when the script is run by path because Python puts the script's own folder on `sys.path`. It:

1. tokenizes `basic-data.txt`, builds the vocabulary, round-trips it through `SimpleTokenizer` and asserts equality with the token list;
2. encodes `"Hello " + raw_text` with `SimpleTokenizerV2` to show the unknown-word path, then decodes it.

## `test_byte_pair_encoding.py`

Introduces the tokenizer everything else uses:

```python
tokenizer = tiktoken.get_encoding("cl100k_base")
```

It encodes `basic-data.txt`, skips the first 50 token ids, and with `context_size = 9` prints:

- `x`, the first 9 ids, and `y`, the same window shifted right by one;
- each growing context with its next token, first as ids and then decoded to text.

This is the input/target relationship that the datasets in [02-data-pipeline.md](02-data-pipeline.md) implement. The file has a `main()` guard.

## Facts about `cl100k_base`

Verified with tiktoken 0.14.0:

| Property | Value |
|---|---|
| `n_vocab` | 100,277 |
| Highest token id | 100,276 |
| `<\|endoftext\|>` id | 100,257 |
| Tokens in `basic-data.txt` | 4,944 (versus 4,690 regex tokens) |
| Ids below `n_vocab` that cannot be decoded | 16: 100,256 and 100,261 to 100,275 |

The model config uses `vocab_size = 100352`, which is 75 more than `n_vocab`. See [04-model.md](04-model.md#the-config).

`tokenizer.decode` raises `KeyError` for any of those 16 ids and for the 75 above `n_vocab`. The model can sample all 91; see [known issue 11](08-known-issues.md#11-sampling-an-id-the-tokenizer-cannot-decode-stops-the-run).

## How special tokens are handled across the repo

The call sites do not agree on `allowed_special`:

| Call site | Argument | Effect on a literal `<\|endoftext\|>` in the text |
|---|---|---|
| `CustomDataset` (`tokenizer.encode(txt)`) | none | tiktoken raises `ValueError` |
| `text_to_token_ids` | `{'<\|endoftext\|>'}` | encoded as id 100,257 |
| `prepare_wikitext.py`, `generate.py` | `"all"` | any special-token string in the text becomes its special id |

Verified with tiktoken 0.14.0. The first row means `create_dataloader` fails on any text that contains a special-token string.
