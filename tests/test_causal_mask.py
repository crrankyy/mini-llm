import torch

from mini_llm.model.attention import MultiHeadedAttention
from mini_llm.model.gpt import GPTModel

CONFIG = {
    "vocab_size": 50,
    "context_length": 8,
    "embedding_dimension": 16,
    "n_heads": 2,
    "n_layers": 2,
    "drop_rate": 0.0,
    "qkv_bias": False
}

# a mask that is not applied moves earlier outputs by about 0.1
TOLERANCE = 1e-6


def test_logits_do_not_depend_on_later_tokens():
    torch.manual_seed(0)
    model = GPTModel(CONFIG).eval()
    tokens = torch.randint(0, CONFIG["vocab_size"], (1, CONFIG["context_length"]))

    with torch.no_grad():
        logits = model(tokens)

        for position in range(1, tokens.shape[1]):
            changed_tokens = tokens.clone()
            changed_tokens[0, position] = (tokens[0, position] + 1) % CONFIG["vocab_size"]
            changed_logits = model(changed_tokens)

            assert torch.allclose(
                changed_logits[:, :position], logits[:, :position], atol=TOLERANCE
            )
            assert not torch.allclose(
                changed_logits[:, position], logits[:, position], atol=TOLERANCE
            )


def test_attention_does_not_read_later_positions():
    torch.manual_seed(0)
    attention = MultiHeadedAttention(
        d_in=16, d_out=16, dropout=0.0, num_heads=2
    ).eval()
    x = torch.randn(1, 8, 16)

    with torch.no_grad():
        context_vectors = attention(x)

        for position in range(1, x.shape[1]):
            changed_x = x.clone()
            changed_x[0, position] += 1.0

            assert torch.allclose(
                attention(changed_x)[:, :position], context_vectors[:, :position], atol=TOLERANCE
            )
