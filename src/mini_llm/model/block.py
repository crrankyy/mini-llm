import torch.nn as nn

from mini_llm.model.attention import MultiHeadedAttention
from mini_llm.model.layers import LayerNorm, FeedForward
    

class TransformerBlock(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.attention = MultiHeadedAttention(
            d_in = config["embedding_dimension"],
            d_out = config["embedding_dimension"],
            context_length = config["context_length"],
            num_heads = config["n_heads"],
            dropout = config["drop_rate"],
            qkvbias = config["qkv_bias"]
        )
        self.ff = FeedForward(config)
        self.norm1 = LayerNorm(config["embedding_dimension"])
        self.norm2 = LayerNorm(config["embedding_dimension"])
        self.drop_shortcut = nn.Dropout(config["drop_rate"])

    def forward(self, x):
        
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x