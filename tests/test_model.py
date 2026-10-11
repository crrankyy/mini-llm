import math

import torch

from mini_llm.model.attention import MultiHeadedAttention
from mini_llm.model.gpt import GPTModel
from mini_llm.model.layers import GELU, LayerNorm

CONFIG = {
    "vocab_size": 1000,
    "context_length": 8,
    "embedding_dimension": 16,
    "n_heads": 2,
    "n_layers": 2,
    "drop_rate": 0.0,
    "qkv_bias": False
}

# the layers call PyTorch's own functions; these tests hold them to the maths written out by hand
TOLERANCE = 1e-6


def test_the_output_layer_and_the_token_embedding_are_one_matrix():
    model = GPTModel(CONFIG)

    assert model.out_head.weight is model.token_embedding.weight
    assert [name for name, _ in model.named_parameters() if name.startswith("out_head")] == []

    # loading weights copies values into the matrix and leaves the two layers sharing it
    model.load_state_dict(GPTModel(CONFIG).state_dict())

    assert model.out_head.weight is model.token_embedding.weight


def test_a_new_model_scores_close_to_a_uniform_guess():
    torch.manual_seed(0)
    model = GPTModel(CONFIG).eval()
    tokens = torch.randint(0, CONFIG["vocab_size"], (4, CONFIG["context_length"] + 1))

    with torch.no_grad():
        logits = model(tokens[:, :-1])
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten())

    assert abs(loss.item() - math.log(CONFIG["vocab_size"])) < 0.05


def test_layer_norm_matches_the_formula():
    torch.manual_seed(0)
    norm = LayerNorm(16)
    with torch.no_grad():
        norm.scale.normal_()
        norm.shift.normal_()
    x = torch.randn(2, 8, 16)

    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    expected = norm.scale * (x - mean) / torch.sqrt(var + norm.eps) + norm.shift

    assert torch.allclose(norm(x), expected, atol=TOLERANCE)


def test_gelu_matches_the_tanh_formula():
    x = torch.linspace(-6, 6, 241)

    expected = 0.5 * x * (1 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))

    assert torch.allclose(GELU()(x), expected, atol=TOLERANCE)


def test_attention_matches_the_steps_written_out():
    torch.manual_seed(0)
    attention = MultiHeadedAttention(d_in=16, d_out=16, dropout=0.0, num_heads=2).eval()
    x = torch.randn(3, 8, 16)

    def heads(tensor):
        return tensor.view(3, 8, 2, 8).transpose(1, 2)

    with torch.no_grad():
        queries, keys, values = heads(attention.W_query(x)), heads(attention.W_key(x)), heads(attention.W_value(x))
        scores = queries @ keys.transpose(2, 3)
        later_positions = torch.triu(torch.ones(8, 8), diagonal=1).bool()
        weights = torch.softmax(scores.masked_fill(later_positions, -torch.inf) / 8 ** 0.5, dim=-1)
        expected = attention.out_proj((weights @ values).transpose(1, 2).reshape(3, 8, 16))

        assert torch.allclose(attention(x), expected, atol=TOLERANCE)


def test_attention_dropout_is_applied_in_training_only():
    torch.manual_seed(0)
    attention = MultiHeadedAttention(d_in=16, d_out=16, dropout=0.5, num_heads=2)
    x = torch.randn(1, 8, 16)

    with torch.no_grad():
        assert not torch.equal(attention(x), attention(x))

        attention.eval()

        assert torch.equal(attention(x), attention(x))
