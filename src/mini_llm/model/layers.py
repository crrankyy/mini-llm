import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    
    def __init__(self, embedding_dimension):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(embedding_dimension))
        self.shift = nn.Parameter(torch.zeros(embedding_dimension))

    def forward(self, x):
        return F.layer_norm(x, self.scale.shape, self.scale, self.shift, self.eps)


class GELU(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.gelu(x, approximate="tanh")
    

class FeedForward(nn.Module):
    
    def __init__(self, config):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(
                config["embedding_dimension"], 4*config["embedding_dimension"]
            ),
            GELU(),
            nn.Linear(
                4*config["embedding_dimension"], config["embedding_dimension"]
            )
        )

    def forward(self, x):
        return self.layers(x)