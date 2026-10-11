import tiktoken
import torch

from mini_llm.generation import decode_token_ids, generate_and_print_sample
from mini_llm.model.gpt import GPTModel

# the real vocabulary size, so the model has output ids the tokenizer has no token for
CONFIG = {
    "vocab_size": 100352,
    "context_length": 8,
    "embedding_dimension": 8,
    "n_heads": 2,
    "n_layers": 1,
    "drop_rate": 0.0,
    "qkv_bias": False
}

# ids below this all have a token in cl100k_base
FIRST_ID_WITHOUT_A_TOKEN = 100256
ID_WITHOUT_A_TOKEN = 100289


def test_decoding_matches_the_tokenizer_when_every_id_has_a_token():
    tokenizer = tiktoken.get_encoding("cl100k_base")
    torch.manual_seed(0)

    # random ids cut multi-byte characters apart, which is where decoding token by token could differ
    for _ in range(200):
        token_ids = torch.randint(0, FIRST_ID_WITHOUT_A_TOKEN, (40,)).tolist()

        assert decode_token_ids(token_ids, tokenizer) == tokenizer.decode(token_ids)


def test_an_id_without_a_token_is_shown_as_its_number():
    tokenizer = tiktoken.get_encoding("cl100k_base")
    token_ids = tokenizer.encode("Hello") + [ID_WITHOUT_A_TOKEN] + tokenizer.encode(" world")

    assert decode_token_ids(token_ids, tokenizer) == "Hello<|100289|> world"


def test_the_training_sample_survives_an_id_without_a_token(capsys):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    torch.manual_seed(0)
    model = GPTModel(CONFIG)

    # every position gets the same hidden vector and only one output row is not zero,
    # so the sampler picks that id every time
    with torch.no_grad():
        model.final_norm.scale.zero_()
        model.final_norm.shift.fill_(1.0)
        model.out_head.weight.zero_()
        model.out_head.weight[ID_WITHOUT_A_TOKEN] = 10.0

    generate_and_print_sample(model, tokenizer, torch.device("cpu"), "Hello")

    assert "Hello<|100289|><|100289|>" in capsys.readouterr().out
