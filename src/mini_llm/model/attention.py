import torch
import torch.nn as nn


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkvbias=False):
        super().__init__()

        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        
        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkvbias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkvbias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkvbias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask", 
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        batch_size, num_tokens, d_in = x.size()
        assert d_in == self.d_in, f"Expected input dimension {self.d_in}, but got {d_in}"

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        )
        queries = queries.view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        )
        values = values.view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        )

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attention_scores = torch.matmul(queries, keys.transpose(2, 3))

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attention_scores = attention_scores.masked_fill(mask_bool, -torch.inf)

        attention_weights = torch.softmax(
            attention_scores/keys.shape[-1]**0.5, dim=-1
        )

        attention_weights = self.dropout(attention_weights)

        context_vectors = (torch.matmul(attention_weights, values)).transpose(1, 2)

        context_vectors = context_vectors.contiguous().view(
            batch_size, num_tokens, self.d_out
        )

        context_vectors = self.out_proj(context_vectors)

        return context_vectors