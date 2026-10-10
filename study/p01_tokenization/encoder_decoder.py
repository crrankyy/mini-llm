import re

def tokenize(text):
    tokens = re.split(r'([,.;:!?_"()\']|--|\s)', text)
    tokens = [token for token in tokens if token.strip()]
    return tokens

def vocabulary_builder(tokens):
    unique_tokens = sorted(set(tokens))
    unique_tokens.extend(["<|unk|>", "<|endoftext|>"])
    vocab = {token:integer for integer, token in enumerate(unique_tokens)}
    return vocab

class SimpleTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def encoder(self, text):
        tokens = tokenize(text)
        encoded_text = [self.vocab.get(token) for token in tokens]
        return encoded_text

    def decoder(self, encoded_text):
        decoded_text = [list(self.vocab.keys())[list(self.vocab.values()).index(i)] for i in encoded_text]
        return decoded_text
    

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.vocab = vocab
        self.inverse_vocab = {v: k for k, v in vocab.items()}

    def encoder(self, text):
        tokens = tokenize(text)
        encoded_text = [self.vocab[token] if token in self.vocab else "<|unk|>" for token in tokens]
        return encoded_text
    
    def decoder(self, encoded_text):
        decoded_text = [self.inverse_vocab[i] if i in self.inverse_vocab else "<|unk|>" for i in encoded_text]
        text = " ".join(decoded_text)
        text = re.sub(r'\s([,.;:!?_"()\']|--)', r'\1', text)
        return text