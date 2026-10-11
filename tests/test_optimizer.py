import torch

from mini_llm.model.gpt import GPTModel
from mini_llm.training.optimizer import create_optimizer

CONFIG = {
    "vocab_size": 50,
    "context_length": 8,
    "embedding_dimension": 16,
    "n_heads": 2,
    "n_layers": 2,
    "drop_rate": 0.0,
    "qkv_bias": False
}

LR, WEIGHT_DECAY = 0.1, 0.5


def test_a_step_without_gradients_shrinks_the_matrices_and_leaves_the_vectors_alone():
    torch.manual_seed(0)
    model = GPTModel(CONFIG)
    with torch.no_grad():
        # biases and shifts start at zero, where a decay would not show
        for parameter in model.parameters():
            if parameter.dim() < 2:
                parameter.normal_()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = create_optimizer(model, lr=LR, weight_decay=WEIGHT_DECAY)

    # with every gradient at zero, the decay is the only thing a step does
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()

    for name, parameter in model.named_parameters():
        if parameter.dim() >= 2:
            assert torch.allclose(parameter, before[name] * (1 - LR * WEIGHT_DECAY), atol=1e-7), name
        else:
            assert torch.equal(parameter, before[name]), name


def test_every_parameter_is_in_one_group_and_the_tied_matrix_is_decayed():
    model = GPTModel(CONFIG)
    optimizer = create_optimizer(model, lr=LR, weight_decay=WEIGHT_DECAY)
    decayed, not_decayed = optimizer.param_groups

    grouped = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]

    assert sorted(grouped) == sorted(id(parameter) for parameter in model.parameters())
    assert sum(parameter is model.out_head.weight for parameter in decayed["params"]) == 1
    assert all(parameter.dim() == 1 for parameter in not_decayed["params"])


def test_both_groups_start_at_the_given_rate_with_the_betas_of_the_published_runs():
    optimizer = create_optimizer(GPTModel(CONFIG), lr=LR, weight_decay=WEIGHT_DECAY)

    assert [group["lr"] for group in optimizer.param_groups] == [LR, LR]
    assert [group["betas"] for group in optimizer.param_groups] == [(0.9, 0.95), (0.9, 0.95)]
