import tiktoken
import torch


raw_text = "The rivers danced under the moonlight as the stars whispered secrets to the sky."

tokenizer = tiktoken.get_encoding("cl100k_base")

encoded_text = tokenizer.encode(raw_text)

vocab_size = tokenizer.n_vocab
output_dim = 3

torch.manual_seed(123)

embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

embedded_text = embedding_layer(torch.tensor(encoded_text))

print("Raw Text:", raw_text)
print("-" * 100)
print("Embedded Text:", embedded_text)
print("-" * 100)

query = embedded_text[5]

print("Query:", query)
print ("Queried word:", tokenizer.decode([encoded_text[5]]))
print("-" * 100)

attn_scores = torch.empty((embedded_text.shape[0]))

for i, word_embedding in enumerate(embedded_text):
    attn_scores[i] = torch.dot(query, word_embedding)

print("Attention Scores:", attn_scores)
print("-" * 100)

normalized_attn_scores = torch.nn.functional.softmax(attn_scores, dim=0)
print("Normalized Attention Scores:", normalized_attn_scores)
print("Sum of Normalized Attention Scores:", normalized_attn_scores.sum())
print("-" * 100)

context_vector = torch.zeros(output_dim)

for i, word_embedding in enumerate(embedded_text):
    context_vector += normalized_attn_scores[i] * word_embedding

print("Context Vector:", context_vector)
print("-" * 100)

# Compute all context vectors for each word in the embedded text
context_vectors = torch.zeros((embedded_text.shape[0], output_dim))

for i, query in enumerate(embedded_text):
    attn_scores = torch.empty((embedded_text.shape[0]))
    
    for j, word_embedding in enumerate(embedded_text):
        attn_scores[j] = torch.dot(query, word_embedding)
    
    normalized_attn_scores = torch.nn.functional.softmax(attn_scores, dim=0)
    
    context_vector = torch.zeros(output_dim)
    
    for j, word_embedding in enumerate(embedded_text):
        context_vector += normalized_attn_scores[j] * word_embedding
    
    context_vectors[i] = context_vector

print("Context Vectors for Each Word:", context_vectors)
print("-" * 100)
