import torch


def create_optimizer(model, lr, weight_decay):

    # Weight decay shrinks a parameter a little at every update, which holds the size of
    # the matrices, the embeddings among them, in check. Biases and the scale and shift of
    # the normalisation layers are left alone: they are 0.06% of the full model's
    # parameters, and decay would pull a scale towards 0 when the value that leaves its
    # input unchanged is 1
    decayed = [p for p in model.parameters() if p.dim() >= 2]
    not_decayed = [p for p in model.parameters() if p.dim() < 2]

    # The second beta is 0.95 where PyTorch's default is 0.999, as in the published runs
    # the learning rate is taken from. The average of squared gradients then covers about
    # the last 20 updates and not the last 1,000, so it follows a change in their size sooner
    return torch.optim.AdamW(
        [
            {"params": decayed, "weight_decay": weight_decay},
            {"params": not_decayed, "weight_decay": 0.0}
        ],
        lr=lr, betas=(0.9, 0.95)
    )
