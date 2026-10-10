# 04: Model

The model is assembled from three files: the building blocks in `src/mini_llm/model/layers.py`, the block in `src/mini_llm/model/block.py`, and the full network in `src/mini_llm/model/gpt.py`. Attention is covered in [03-attention.md](03-attention.md).

## The config

`src/mini_llm/config.py` holds the only config used by the entry points:

```python
GPT_CONFIG_124M = {
    "vocab_size": 100352,
    "context_length": 256,
    "embedding_dimension": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
```

| Key | Used by | Notes |
|---|---|---|
| `vocab_size` | token embedding rows, output head width | `cl100k_base` has 100,277 tokens, so ids 100,277 to 100,351 are never produced by the tokenizer. 100,352 equals 1,568 x 64; the repo does not record why this value was chosen. |
| `context_length` | positional embedding rows, attention mask size, window length in both training scripts | The longest sequence the model accepts. |
| `embedding_dimension` | every layer width | Must be divisible by `n_heads`. |
| `n_heads` | `MultiHeadedAttention` | 12 heads of 64 dimensions. |
| `n_layers` | number of `TransformerBlock`s | |
| `drop_rate` | embedding dropout, attention-weight dropout, shortcut dropout | One rate for all three. |
| `qkv_bias` | query/key/value `nn.Linear` layers | |

`study/p04_gpt/architecture.py` defines a second dict with the same name but `vocab_size = 100277` and `context_length = 1024`. It is local to that sandbox file.

## Building blocks (`src/mini_llm/model/layers.py`)

### `LayerNorm(embedding_dimension)`

Normalizes over the last dimension using the biased variance (`unbiased=False`) and `eps = 1e-5`, then applies a learned `scale` (initialized to ones) and `shift` (initialized to zeros):

```python
norm_x = (x - mean) / torch.sqrt(var + self.eps)
return self.scale * norm_x + self.shift
```

### `GELU()`

The tanh approximation:

```python
0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))
```

### `FeedForward(config)`

`Linear(D, 4D)`, `GELU`, `Linear(4D, D)`. With `D = 768` the hidden width is 3,072. It is applied to each position independently.

## `TransformerBlock(config)` (`src/mini_llm/model/block.py`)

A pre-norm block: normalization comes before each sub-layer, and each sub-layer's output is added back to its input.

```python
shortcut = x
x = self.norm1(x)
x = self.attention(x)
x = self.drop_shortcut(x)
x = x + shortcut

shortcut = x
x = self.norm2(x)
x = self.ff(x)
x = self.drop_shortcut(x)
x = x + shortcut
```

Input and output are both `(B, T, D)`. One `nn.Dropout` instance, `drop_shortcut`, is used for both sub-layers; dropout has no state, so this behaves the same as two instances.

## `GPTModel(config)` (`src/mini_llm/model/gpt.py`)

| Attribute | Layer | Output shape |
|---|---|---|
| `token_embedding` | `nn.Embedding(V, D)` | `(B, T, D)` |
| `position_embedding` | `nn.Embedding(context_length, D)`, learned absolute positions | `(T, D)`, broadcast over the batch |
| `drop_embedding` | `nn.Dropout(drop_rate)` | `(B, T, D)` |
| `transformer_blocks` | `nn.Sequential` of `n_layers` `TransformerBlock`s | `(B, T, D)` |
| `final_norm` | `LayerNorm(D)` | `(B, T, D)` |
| `out_head` | `nn.Linear(D, V, bias=False)` | `(B, T, V)` |

`forward(in_idx)` takes token ids of shape `(B, T)` and returns logits of shape `(B, T, V)`: one score per vocabulary entry at every position.

```python
in_idx = in_idx.to(torch.int32)
token_embedding = self.token_embedding(in_idx)
position_embedding = self.position_embedding(torch.arange(seq_len, device=in_idx.device))
x = token_embedding + position_embedding
x = self.drop_embedding(x)
x = self.transformer_blocks(x)
x = self.final_norm(x)
logits = self.out_head(x)
```

Points worth knowing:

- **`T` must not exceed `context_length`.** A longer input indexes past the end of the positional table. The generation functions crop their input to the last `context_size` tokens for this reason.
- **The ids are cast to `int32`** before the embedding lookup.
- **No weight tying.** `out_head.weight` and `token_embedding.weight` are separate tensors (verified).
- **No custom initialization.** Every layer keeps PyTorch's default init.
- **No softmax in the model.** It returns raw logits; the loss and the samplers apply softmax themselves.

## Parameter budget

Verified by instantiating `GPTModel(GPT_CONFIG_124M)`:

| Component | Parameters |
|---|---|
| `token_embedding` (100,352 x 768) | 77,070,336 |
| `position_embedding` (256 x 768) | 196,608 |
| `transformer_blocks` (12 x 7,085,568) | 85,026,816 |
| `final_norm` | 1,536 |
| `out_head` (768 x 100,352) | 77,070,336 |
| **Total** | **239,365,632** |

One block is 7,085,568 parameters: attention 2,360,064, feed-forward 4,722,432, and 1,536 for each of the two norms.

At 4 bytes per parameter that is 913.11 MB, which is the size `train_tiny.py` and `train_wikitext.py` print. The state dict has 173 entries: 161 parameter tensors and the 12 attention-mask buffers.

**Why this is not 124M.** The two vocabulary-sized matrices account for 154.1M of the total. GPT-2 small reaches 124M with a 50,257-token vocabulary and an output head that shares the embedding matrix. This model has a vocabulary twice that size and does not share the matrix. The transformer stack itself, 85.0M, is the same size as GPT-2 small's apart from the missing query/key/value biases.

## The sandbox: `study/p04_gpt/architecture.py`

A demo script for stage 4. It defines:

- **`DummyGPT(config)`**: the same layers and forward pass as `GPTModel`, without the `int32` cast.
- **`ExampleDeepNeuralNetwork(layer_sizes, use_shortcut)`**: five `Linear` + `GELU` layers. With `use_shortcut=True`, a layer's output is added to its input whenever the shapes match.
- **`print_gradients(model, x)`**: runs one forward and backward pass against a target of 0 with `nn.MSELoss` and prints the mean absolute gradient of each weight matrix.

The module-level code then runs a two-sentence batch through `DummyGPT`, normalizes a small tensor by hand and with `LayerNorm`, checks the `FeedForward` output shape, compares gradients with and without shortcuts, and runs one `TransformerBlock`.

It imports `LayerNorm`, `GELU`, `FeedForward` and `TransformerBlock` from the library, so it needs the package installed. Verified: `python study/p04_gpt/architecture.py` exits with code 0 and ends with `Output shape: torch.Size([2, 4, 768])`. Its `DummyGPT` is built at full size with a 1,024-token context, so the script allocates about 1 GB.

Before the reorganization the script could not start: it imported from a module named `sfeedforward`, which never existed.
