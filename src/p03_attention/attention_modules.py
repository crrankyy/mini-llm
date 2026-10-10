import torch
import torch.nn as nn


class SelfAttention_v1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.W_query  = nn.Parameter(torch.rand(input_dim, output_dim), requires_grad=True)
        self.W_key    = nn.Parameter(torch.rand(input_dim, output_dim), requires_grad=True)
        self.W_value  = nn.Parameter(torch.rand(input_dim, output_dim), requires_grad=True)

    def forward(self, x):
        keys = torch.matmul(x, self.W_key)
        queries = torch.matmul(x, self.W_query)
        values = torch.matmul(x, self.W_value)

        attention_scores = torch.matmul(queries, keys.T)
        d_k = keys.shape[-1]
        attention_weights = torch.softmax(attention_scores / torch.sqrt(torch.tensor(d_k)), dim=-1)
        context_vectors = torch.matmul(attention_weights, values)
        return context_vectors
    
class SelfAttention_v2(nn.Module):
    """
        This version uses Linear layers instead of manually defined weight matrices.
        It also includes dropout and causal masking to prevent attending to future tokens.
    """

    def __init__(self, input_dim, output_dim, qkv_bias=False):
        super().__init__()
        self.W_query  = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_key    = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_value  = nn.Linear(input_dim, output_dim, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attention_scores = torch.matmul(queries, keys.T)
        
        d_k = keys.shape[-1]
        attention_weights = torch.softmax(attention_scores / torch.sqrt(torch.tensor(d_k)), dim=-1)
        
        context_length = attention_scores.shape[0]
        mask = torch.tril(torch.ones(context_length, context_length))
        attention_weights = attention_weights * mask
        
        row_sums = attention_weights.sum(dim=-1, keepdim=True)
        attention_weights = attention_weights / row_sums

        dropout = nn.Dropout(p=0.1)
        attention_weights = dropout(attention_weights)

        context_vectors = torch.matmul(attention_weights, values)
        return context_vectors
    

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, num_heads, qkvbias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [SelfAttention_v2(
                d_in, d_out, qkv_bias=qkvbias
                )
                for _ in range(num_heads)
            ]
        )

    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]
        concatenated = torch.cat(head_outputs, dim=-1)
        return concatenated
    

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