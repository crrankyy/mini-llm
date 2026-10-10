import tiktoken
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
    

raw_text = "The rivers danced under the moonlight as the stars whispered secrets to the sky."

tokenizer = tiktoken.get_encoding("cl100k_base")

encoded_text = tokenizer.encode(raw_text)

vocab_size = tokenizer.n_vocab
output_dim = 5

torch.manual_seed(123)

embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

embedded_text = embedding_layer(torch.tensor(encoded_text))

print("Raw Text:", raw_text)
print("Length of Raw Text:", len(raw_text))
print("-" * 100)
# print("Embedded Text:", embedded_text)
print("Size of Embedded Text:", embedded_text.shape)
print("-" * 100)

query_n = embedded_text[5]
d_in = embedded_text.shape[1]

W_query = torch.nn.Parameter(torch.randn(d_in, output_dim), requires_grad=True)
W_key = torch.nn.Parameter(torch.randn(d_in, output_dim), requires_grad=True)
W_value = torch.nn.Parameter(torch.randn(d_in, output_dim), requires_grad=True)

query = torch.matmul(query_n, W_query)
key = torch.matmul(query_n, W_key)
value = torch.matmul(query_n, W_value)

print("Query:", query)
print("Key:", key)
print("Value:", value)
print ("Queried word:", tokenizer.decode([encoded_text[5]]))
print("-" * 100)
print("-" * 100)
print("-" * 100)

queries = torch.matmul(embedded_text, W_query)
keys = torch.matmul(embedded_text, W_key)
values = torch.matmul(embedded_text, W_value)

print("Size of all Keys:", keys.shape)
print("Size of all Values:", values.shape)
print("-" * 100)
print("-" * 100)

key_2 = keys[1]
value_2 = values[1]
query_2 = queries[1]

attn_score_22 = torch.dot(query_2, key_2)
print("Attention Score for Query 2 and Key 2:", attn_score_22)
print("-" * 100)
attn_scores = torch.matmul(queries, keys.T)
print("Attention Scores for Query 2 and all Keys:", attn_scores[1])
print("-" * 100)
print("-" * 100)

print("Size of Attention Scores:", attn_scores.shape)
print("-" * 100)
print("-" * 100)

d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores[1] / torch.sqrt(torch.tensor(d_k)), dim=-1)
print("Attention Weights for Query 2:", attn_weights_2)
print("Sum of Attention Weights for Query 2:", attn_weights_2.sum())
print("-" * 100)
print("-" * 100)

context_vector_2 = torch.matmul(attn_weights_2, values)
print("Context Vector for Query 2:", context_vector_2)
print("-" * 100)
print("-" * 100)

context_vectors = torch.matmul(torch.softmax(attn_scores / torch.sqrt(torch.tensor(d_k)), dim=-1), values)
print("Size of Context Vectors:", context_vectors.shape)
print("-" * 100)
print("-" * 100)    
print("-" * 100)    
print("-" * 100)    

sa_v1 = SelfAttention_v1(input_dim=output_dim, output_dim=output_dim)
print(sa_v1(embedded_text).shape)

sa_v2 = SelfAttention_v2(input_dim=output_dim, output_dim=output_dim)
print(sa_v2(embedded_text).shape)

mha = MultiHeadAttentionWrapper(d_in=output_dim, d_out=output_dim, num_heads=3)
print(mha(embedded_text).shape)