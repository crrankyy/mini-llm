# 03: Attention

Stage 3 lives in `src/p03_attention/`. It goes from attention with no trainable weights to the multi-head module that the model uses. Only `MultiHeadedAttention` from `attention_modules.py` is imported outside this folder.

## The three files

| File | Role |
|---|---|
| `simple_attention.py` | Demo script: dot-product attention over raw embeddings |
| `self_attention.py` | Demo script: scaled dot-product attention with query/key/value matrices, followed by the four classes |
| `attention_modules.py` | The four classes on their own, importable without running a demo |

The class definitions in `self_attention.py` and `attention_modules.py` are identical copies.

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

The module `TransformerBlock` uses. All heads share one set of projection layers and are computed in a single batched matrix product. `d_out` must be divisible by `num_heads`; each head gets `head_dim = d_out // num_heads` dimensions.

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

## The causal mask

Step 5 replaces every score above the diagonal with `-inf`, so softmax gives those positions a weight of zero and each token attends only to itself and the tokens before it. `masked_fill` returns a new tensor and leaves its input unchanged, which is why the result is assigned back.

Verified on 2026-10-11: in a small `GPTModel` in eval mode with dropout 0, changing the token at position `t` leaves the logits at every earlier position exactly unchanged, for every `t`.

Until that date the return value was discarded and the mask had no effect. Weights trained before the fix need retraining. See [08-known-issues.md](08-known-issues.md#1-the-causal-mask-was-never-applied-fixed).

`SelfAttention_v2` masks differently: after softmax, by multiplication.

## What carries over into checkpoints

Because `mask` is a registered buffer, each of the 12 blocks stores a `256 x 256` mask in the state dict. A saved model is therefore tied to its `context_length` twice: through the mask shape and through the positional embedding table.
