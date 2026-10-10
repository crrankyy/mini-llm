import re

def tokenize(text):
    tokens = re.split(r'([,.;:!?_"()\']|--|\s)', text)
    tokens = [token for token in tokens if token.strip()]
    return tokens

def vocabulary_builder(tokens):
    unique_tokens = sorted(set(tokens))
    vocab = {token:integer for integer, token in enumerate(unique_tokens)}
    return vocab

def encoder(text, vocab):
    tokens = tokenize(text)
    encoded_text = [vocab[token] for token in tokens]
    return encoded_text, vocab

def decoder(encoded_text, vocab):
    reverse_vocab = {integer: token for token, integer in vocab.items()}
    decoded_text = [reverse_vocab[integer] for integer in encoded_text]
    return decoded_text

with open("data/basic-data.txt", "r") as f:
    raw_text = f.read()


print(f"-" * 30)

print(f"Total number of characters: {len(raw_text)}")
print(f"Text:\n{raw_text[:100]}...")
print(f"-" * 30)

# Tokenization
tokens = tokenize(raw_text)

print(f"Total number of tokens: {len(tokens)}")
print(f"Tokens:\n{tokens[:100]}...") 
print(f"-" * 30)

# Convert tokens to integers
vocab = vocabulary_builder(tokens)
encoded_text, vocab = encoder(raw_text, vocab)

print(f"Encoded text (integer IDs):\n{encoded_text[:100]}...")
print(f"-" * 30)

# Decode the encoded text
decoded_text = decoder(encoded_text, vocab)
print(f"Decoded text:\n{decoded_text[:100]}...")
print(f"-" * 30)

# Check if the decoded text matches the original text
assert decoded_text == tokens, "Decoded text does not match the original text"
print("Decoded text matches the original text!")