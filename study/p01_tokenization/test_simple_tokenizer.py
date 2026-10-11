#import tokenizer functions
import encoder_decoder as ed

with open("data/basic-data.txt", "r") as f:
    raw_text = f.read()


print(f"-" * 30)
print(f"A very simple tokenizer and encoder/decoder implementation")
print(f"-" * 30)
print(f"Total number of characters: {len(raw_text)}")
print(f"Text:\n{raw_text[:100]}...")
print(f"-" * 30)    

# Tokenization
tokens = ed.tokenize(raw_text)
print(f"Total number of tokens: {len(tokens)}")
print(f"Tokens:\n{tokens[:100]}...")
print(f"-" * 30)

# Convert tokens to integers
vocab = ed.vocabulary_builder(tokens)
tokenizer = ed.SimpleTokenizer(vocab)
encoded_text = tokenizer.encoder(raw_text)
print(f"Encoded text (integer IDs):\n{encoded_text[:100]}...")
print(f"-" * 30)

# Decode the encoded text
decoded_text = tokenizer.decoder(encoded_text)
print(f"Decoded text:\n{decoded_text[:100]}...")
print(f"-" * 30)

# Check if the decoded text matches the original text
assert decoded_text == tokens, "Decoded text does not match the original text"
print("Decoded text matches the original text!")
print(f"-" * 30)
print(f"-" * 30)

print(f"A more advanced tokenizer and encoder/decoder implementation with token for unknown words")
print(f"-" * 30)
# Convert tokens to integers using SimpleTokenizerV2
tokenizer_v2 = ed.SimpleTokenizerV2(vocab)
encoded_text_v2 = tokenizer_v2.encoder("Hello "+raw_text)
print(f"Encoded text (with unknown token handling):\n{encoded_text_v2[:100]}...")
print(f"-" * 30)

# Decode the encoded text using SimpleTokenizerV2
decoded_text_v2 = tokenizer_v2.decoder(encoded_text_v2)
print(f"Decoded text (with unknown token handling):\n{decoded_text_v2[:100]}...")
print(f"-" * 30)


