# 06: Generation and inference

The model predicts one token at a time. Every generation function here runs the same loop: feed the sequence in, take the logits at the last position, choose a token, append it, repeat. Nothing is cached between steps, so each new token costs a full forward pass over the whole window.

## Generation functions

### `textgen_v1(model, idx, max_new_tokens, context_size)`

Greedy decoding, in `src/mini_llm/generation.py`.

```python
for _ in range(max_new_tokens):
    idx_cond = idx[:, -context_size:]
    with torch.no_grad():
        logits = model(idx_cond)
    logits = logits[:, -1, :]
    probs = torch.softmax(logits, dim=-1)
    idx_next = torch.argmax(probs, dim=-1, keepdim=True)
    idx = torch.cat((idx, idx_next), dim=1)
return idx
```

- `idx` is `(B, T)`. The return value is `(B, T + max_new_tokens)`: the prompt followed by the new tokens.
- `idx[:, -context_size:]` keeps the input within the model's positional table.
- The softmax does not change which token `argmax` picks.
- It does not call `model.eval()`; the caller is expected to.

### `textgen_v2(model, idx, max_new_tokens, context_size, temperature=1.0, top_k=None)`

In `src/mini_llm/generation.py`. Adds two controls to the same loop:

- **Top-k.** If `top_k` is set, every logit below the k-th largest is replaced with `-inf`, so it gets zero probability. Logits tied with the k-th largest are kept.
- **Temperature.** If `temperature > 0`, logits are divided by it, passed through softmax, and one token is sampled with `torch.multinomial`. Values below 1 sharpen the distribution; values above 1 flatten it. If `temperature` is 0 or negative, it falls back to `argmax`.

Top-k is applied before temperature. There is no stop condition; it always generates `max_new_tokens`.

### Helpers in `src/mini_llm/generation.py`

| Function | Behaviour |
|---|---|
| `text_to_token_ids(text, tokenizer)` | Encodes with `allowed_special={'<\|endoftext\|>'}` and adds a batch dimension, giving shape `(1, T)` |
| `decode_token_ids(token_ids, tokenizer)` | Decodes a list of ids one at a time. An id the tokenizer has no token for is written as its number, for example `<\|100289\|>`; when every id has a token the result equals `tokenizer.decode` (verified by `tests/test_generation.py`). |
| `token_ids_to_text(token_ids, tokenizer)` | Removes the batch dimension and decodes with `decode_token_ids`. Expects a batch of one. |

### `generate_and_print_sample(model, tokenizer, device, start_context)`

Called by the trainer at the end of each epoch. It switches to eval mode, reads the context size from `model.position_embedding.weight.shape[0]`, generates 50 tokens with `textgen_v2(temperature=0.8, top_k=25)`, prints the result with newlines replaced by spaces, and switches back to train mode.

Reading the context size through the attribute works on a compiled model too, because the wrapper forwards attribute access (verified on torch 2.2.2).

## `generate.py`

`scripts/generate.py` is an interactive prompt for a trained model. It has a `main()` guard, so its functions can be imported. It does not use the functions above; it carries its own copy of the sampling loop.

```sh
uv run python scripts/generate.py                                                        # search checkpoints/
uv run python scripts/generate.py --checkpoint-dir some/folder                           # search another folder
uv run python scripts/generate.py --weights checkpoints/gpt_wikitext_checkpoint_ep1.pth  # one specific file
```

### `find_weights(checkpoint_dir)`

Returns `gpt_wikitext_124M.pth` in that folder if it exists. Otherwise it returns the last entry of `sorted(glob.glob("gpt_wikitext_checkpoint_*.pth"))` in the folder, and if there is none it returns the path of the missing final weights, so that the loader reports it.

### `load_model_weights(model, path, device)`

1. Raises `FileNotFoundError` if `path` does not exist.
2. `torch.load(path, map_location=device, weights_only=False)`.
3. If the loaded object is a dict containing `model_state_dict`, uses that entry. This lets it read both trainer checkpoints and the bare state dict in `gpt_wikitext_124M.pth`.
4. Strips `_orig_mod.` from every key, then `model.load_state_dict(...)`.

Verified against a checkpoint written by the trainer.

`map_location=device` moves everything in the file to the device, including the optimizer state of a full checkpoint, which is then dropped. Verified on MPS with a 2.88 GB checkpoint of the full model whose keys carried the `_orig_mod.` prefix: it loaded and generated.

`weights_only=False` means the file is unpickled without restriction, so it can execute code. Only load files you produced yourself.

### `generate_text(model, tokenizer, prompt, device, max_new_tokens=100, temperature=0.7, top_k=50)`

The same top-k and temperature logic as `textgen_v2`, with three differences:

- **It stops early** when the sampled token is `<|endoftext|>` (id 100,257, looked up from the tokenizer with a hardcoded fallback).
- **It returns only the new text** as a string. The prompt is not included (verified).
- **It handles one sequence at a time**, because it calls `idx_next.item()`.

The context size comes from `GPT_CONFIG_124M["context_length"]`, not from the model instance. The prompt is encoded with `allowed_special="all"`.

The new ids are decoded with `decode_token_ids`, so one of the 91 ids the tokenizer has no token for appears in the response as its number; see [known issue 11](08-known-issues.md#11-sampling-an-id-the-tokenizer-cannot-decode-stopped-the-run-fixed).

### `run_cli()`

1. Reads `--weights` and `--checkpoint-dir` (default `checkpoints`).
2. Picks the device with `get_device()`: `cuda`, then `mps`, then `cpu`.
3. Builds `GPTModel(GPT_CONFIG_124M)`.
4. Chooses the weights file: `--weights` if given, otherwise `find_weights(checkpoint_dir)`.
5. Loads the weights. If no file is found, prints an error that names the folder and returns.
6. Loops on `Prompt >>> `. `quit` or `exit` ends the loop, an empty line is skipped, and `Ctrl-C` exits. Each prompt is answered with `generate_text(max_new_tokens=150, temperature=0.8, top_k=40)`.

Two consequences of the search in step 4, both avoided by passing `--weights`:

- The fallback pattern `gpt_wikitext_checkpoint_*.pth` also matches `gpt_wikitext_checkpoint_interrupted.pth`, and that name sorts **after** every `..._ep{N}.pth` (verified). If both exist and the final weights file does not, the interrupted checkpoint is the one loaded.
- `gpt_wikitext_124M.pth` is written by `train_wikitext.py` even after an interrupted run, so its presence does not mean training finished.

Verified at full size on the Intel MacBook's GPU: loading the 2.88 GB checkpoint and generating 150 tokens took 63 seconds in total.

Each prompt is independent. The banner calls it "Interactive Chat/Generation Mode", but there is no conversation history and no chat formatting: the model continues the text it is given.

## Sampling settings in use

| Caller | `max_new_tokens` | `temperature` | `top_k` | Stops at end-of-text |
|---|---|---|---|---|
| `train_tiny.py` via `textgen_v1` | 10 | greedy | none | no |
| Trainer via `generate_and_print_sample` | 50 | 0.8 | 25 | no |
| `generate_text` defaults | 100 | 0.7 | 50 | yes |
| `run_cli` | 150 | 0.8 | 40 | yes |
