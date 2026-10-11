import torch
import torch.nn as nn

from mini_llm.model.layers import LayerNorm
from mini_llm.model.block import TransformerBlock

class GPTModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(
                config["vocab_size"], config["embedding_dimension"]
        )
        self.position_embedding = nn.Embedding(
                config["context_length"], config["embedding_dimension"]
        )
        
        self.drop_embedding = nn.Dropout( config["drop_rate"] )

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(config)
                for _ in range(config["n_layers"])
            ]
        )

        self.final_norm = LayerNorm( config["embedding_dimension"] )

        self.out_head = nn.Linear(
                config["embedding_dimension"], config["vocab_size"], bias = False
        )
        # tied: the matrix that turns a token id into a vector also scores every
        # token against the final vector, so the model stores it once
        self.out_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # PyTorch draws an embedding at std 1. Read back through the tied output layer,
        # weights that large give a first loss in the hundreds, where a uniform guess
        # over the vocabulary would score ln(vocab_size)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        
        in_idx = in_idx.to(torch.int32)
        token_embedding = self.token_embedding(in_idx)
        
        position_embedding = self.position_embedding(
            torch.arange(seq_len, device=in_idx.device)
        )
        
        x = token_embedding + position_embedding
        
        x = self.drop_embedding(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        
        logits = self.out_head(x)
        
        return logits