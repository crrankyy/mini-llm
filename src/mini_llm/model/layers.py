import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    
    def __init__(self, embedding_dimension):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(embedding_dimension))
        self.shift = nn.Parameter(torch.zeros(embedding_dimension))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (
            1 + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi)) *
                    (x + 0.044715 * torch.pow(x, 3))
                )
        )
    

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