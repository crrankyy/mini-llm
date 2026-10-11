import torch
import torch.nn as nn

vocab_size = 8
output_dim = 4

torch.manual_seed(123)

print("-" * 100)

embedding_layer = nn.Embedding(vocab_size, output_dim)
print("Embedding layer weights:")
print("-" * 100)

print(embedding_layer.weight)
print("-" * 100)
print("-" * 100)

print("Input tensor:")
print("-" * 100)
input_tensor = torch.tensor([1, 2, 3, 4])
print(input_tensor)
print("-" * 100)
print("Embedded output:")
print("-" * 100)
embedded_output = embedding_layer(input_tensor)
print(embedded_output)
print("-" * 100)
print("-" * 100)
