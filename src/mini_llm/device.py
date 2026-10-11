import torch

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    # Metal, the only way to reach the GPU on a Mac
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")
