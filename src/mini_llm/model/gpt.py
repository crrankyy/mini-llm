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