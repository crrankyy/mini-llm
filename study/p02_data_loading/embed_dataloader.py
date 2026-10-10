import torch
import torch.nn as nn
import study.p02_data_loading.dataloader as DataLoader

import tiktoken

with open("data/basic-data.txt", "r") as f:
    raw_text = f.read()

tokenizer = tiktoken.get_encoding("cl100k_base")

vocab_size = tokenizer.n_vocab
output_dim = 256
context_length = 128

embedding_layer = nn.Embedding(vocab_size, output_dim)

dataloader = DataLoader.create_dataloader(
                raw_text, 
                batch_size=8, 
                max_length=context_length, 
                stride=(context_length // 2), 
                shuffle=False
            )

data_iter = iter(dataloader)

input_ids, target_ids = next(data_iter)

token_embeddings = embedding_layer(input_ids)

print("-" * 100)
print("Input IDs:", input_ids)
print("Target IDs:", target_ids)
print("Token Embeddings:", token_embeddings)
print("-" * 100)