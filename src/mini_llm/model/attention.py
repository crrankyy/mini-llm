import torch.nn as nn
import torch.nn.functional as F


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_in, d_out, dropout, num_heads, qkvbias=False):
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
        self.dropout = dropout

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

        # scores, causal mask, softmax, dropout and the weighted sum of the values, all in
        # PyTorch's own function. study/p03_attention/self_attention.py has the steps written out
        context_vectors = F.scaled_dot_product_attention(
            queries, keys, values, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0
        ).transpose(1, 2)

        context_vectors = context_vectors.contiguous().view(
            batch_size, num_tokens, self.d_out
        )

        context_vectors = self.out_proj(context_vectors)

        return context_vectors