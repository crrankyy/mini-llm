# 03: Attention

Stage 3 lives in `study/p03_attention/`. It goes from attention with no trainable weights to the multi-head module that the model uses. That module, `MultiHeadedAttention`, is in the library at `src/mini_llm/model/attention.py`.

## The three files

| File | Role |
|---|---|
| `study/p03_attention/simple_attention.py` | Demo script: dot-product attention over raw embeddings |
| `study/p03_attention/self_attention.py` | Demo script: the four classes, then scaled dot-product attention with query/key/value matrices step by step |
| `src/mini_llm/model/attention.py` | `MultiHeadedAttention` on its own, importable without running a demo |

`MultiHeadedAttention` in the library has the same layers as the class in `self_attention.py` and gives the same output from the same weights. It hands steps 4 to 7 of the forward pass below to a PyTorch function and stores no mask; see [The library version](#the-library-version). The three earlier classes exist only in `self_attention.py`.

Both demo scripts work on one sentence, "The rivers danced under the moonlight as the stars whispered secrets to the sky.", which is 16 `cl100k_base` tokens, embedded with a seeded `nn.Embedding`.

## `simple_attention.py`

No trainable weights. With 3-dimensional embeddings and the token at index 5 as the query:

1. score every token by its dot product with the query;
2. normalize the scores with softmax;
3. take the weighted sum of all embeddings to get the query's context vector.

A double loop then repeats this for every token to produce all 16 context vectors.

## `self_attention.py` (demo part)

Uses 5-dimensional embeddings and three random `5 x 5` matrices `W_query`, `W_key`, `W_value`. It prints the projections for one token, then for all tokens, then one attention score, the full `16 x 16` score matrix, and the weights after dividing by `sqrt(d_k)` and applying softmax.

It ends by instantiating three of the classes and printing output shapes (verified): `SelfAttention_v1` gives `(16, 5)`, `SelfAttention_v2` gives `(16, 5)`, and `MultiHeadAttentionWrapper` with 3 heads gives `(16, 15)`.

## The four classes

### `SelfAttention_v1(input_dim, output_dim)`

Query, key and value matrices are `nn.Parameter(torch.rand(...))`. Computes `softmax(Q Kᵀ / sqrt(d_k)) V`. There is no mask and no dropout. It uses `keys.T`, so the input must be 2-D, `(tokens, dim)`, with no batch dimension.

### `SelfAttention_v2(input_dim, output_dim, qkv_bias=False)`

Replaces the raw matrices with `nn.Linear` layers and adds causal masking and dropout:

- Masking is done **after** softmax: the weights are multiplied by a lower-triangular matrix of ones and each row is divided by its sum.
- Dropout is created inside `forward` (`nn.Dropout(p=0.1)`). A freshly built module is in training mode, so this dropout stays active after `model.eval()` (verified: two eval-mode calls on the same input give different outputs), and its rate cannot be configured.
- Like v1 it only accepts 2-D input.

### `MultiHeadAttentionWrapper(d_in, d_out, num_heads, qkvbias=False)`

Holds `num_heads` independent `SelfAttention_v2` modules in an `nn.ModuleList`, runs them one after another, and concatenates their outputs on the last dimension. The output width is `d_out * num_heads`.

### `MultiHeadedAttention(d_in, d_out, context_length, dropout, num_heads, qkvbias=False)`

The class in `self_attention.py`, with every step written out. `TransformerBlock` uses the library version described after it. All heads share one set of projection layers and are computed in a single batched matrix product. `d_out` must be divisible by `num_heads`; each head gets `head_dim = d_out // num_heads` dimensions.

Layers and buffers:

| Name | Type | Notes |
|---|---|---|
| `W_query`, `W_key`, `W_value` | `nn.Linear(d_in, d_out, bias=qkvbias)` | no bias under the repo config |
| `out_proj` | `nn.Linear(d_out, d_out)` | has a bias |
| `dropout` | `nn.Dropout(dropout)` | applied to the attention weights |
| `mask` | buffer, `torch.triu(torch.ones(context_length, context_length), diagonal=1)` | ones above the diagonal; saved in the state dict |

Forward pass, with shapes for the repo config (`D = 768`, `H = 12`, `head_dim = 64`):

| Step | Operation | Shape |
|---|---|---|
| 1 | input `x` | `(B, T, 768)` |
| 2 | `W_key(x)`, `W_query(x)`, `W_value(x)` | `(B, T, 768)` each |
| 3 | `.view(B, T, 12, 64)` then `.transpose(1, 2)` | `(B, 12, T, 64)` |
| 4 | `queries @ keys.transpose(2, 3)` | `(B, 12, T, T)` |
| 5 | `attention_scores = attention_scores.masked_fill(mask_bool, -torch.inf)` | `(B, 12, T, T)`, later positions set to `-inf` |
| 6 | `softmax(scores / 64 ** 0.5, dim=-1)`, then dropout | `(B, 12, T, T)` |
| 7 | `weights @ values`, then `.transpose(1, 2)` | `(B, T, 12, 64)` |
| 8 | `.contiguous().view(B, T, 768)` | `(B, T, 768)` |
| 9 | `out_proj` | `(B, T, 768)` |

The mask is sliced to `[:T, :T]`, so sequences shorter than `context_length` work.

### The library version

`MultiHeadedAttention(d_in, d_out, dropout, num_heads, qkvbias=False)` in `src/mini_llm/model/attention.py` has the same four `nn.Linear` layers under the same names. It differs in three ways:

- **No `context_length` argument and no `mask` buffer.** Nothing about the mask is stored.
- **`dropout` is the rate**, a plain number, not an `nn.Dropout` module.
- **Steps 4 to 7 are one call:**

```python
context_vectors = F.scaled_dot_product_attention(
    queries, keys, values, is_causal=True,
    dropout_p=self.dropout if self.training else 0.0
).transpose(1, 2)
```

`is_causal=True` asks for the mask of step 5. The function applies whatever dropout rate it is given, in training or not, so the rate is passed only while the module is in training mode.

Verified on 2026-10-11: `tests/test_model.py` computes steps 4 to 7 by hand from the module's own weights and gets the module's output to within 1e-6. With the same weights loaded before and after the change, the full model's logits moved by at most 2.3e-6.

The change is for speed. Together with the same change to `LayerNorm` and `GELU` (see [04-model.md](04-model.md#building-blocks-srcmini_llmmodellayerspy)), a training step of the full model at batch 2 x 256 took 12% less time on the Intel MacBook's CPU (3.33 s to 2.92 s, measured). It was not measured on a CUDA GPU.

## The causal mask

Step 5 replaces every score above the diagonal with `-inf`, so softmax gives those positions a weight of zero and each token attends only to itself and the tokens before it. `masked_fill` returns a new tensor and leaves its input unchanged, which is why the result is assigned back.

Verified on 2026-10-11: in a small `GPTModel` in eval mode with dropout 0, changing the token at position `t` leaves the logits at every earlier position exactly unchanged, for every `t`.

Until that date the return value was discarded and the mask had no effect. Weights trained before the fix need retraining. See [08-known-issues.md](08-known-issues.md#1-the-causal-mask-was-never-applied-fixed).

The library version gets the same mask by passing `is_causal=True`, and the check above was repeated after that change. `SelfAttention_v2` masks differently: after softmax, by multiplication.

## What carries over into checkpoints

The library version registers no buffer, so a checkpoint holds nothing of the mask, and a saved model is tied to its `context_length` only through the positional embedding table. That table is why a file written while the context was 256 tokens, before 2026-10-12, no longer loads: `size mismatch for position_embedding.weight` (verified).

Until 2026-10-11 each of the 12 blocks stored a `256 x 256` mask in the state dict. `load_state_dict` refuses a checkpoint from then because of those entries (verified). The class in `self_attention.py` still registers its mask.
