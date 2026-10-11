import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

def prepare_data():
    print("Downloading WikiText-103...")
    # Using Salesforce/wikitext because the default wikitext namespace was deprecated
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    os.makedirs("data/wikitext-103", exist_ok=True)

    for split in ['train', 'validation']:
        print(f"Tokenizing {split} split...")
        split_dataset = dataset[split]
        
        all_tokens = []
        for text in tqdm(split_dataset['text']):
            if text.strip():
                all_tokens.extend(tokenizer.encode(text, allowed_special="all"))
        
        arr = np.array(all_tokens, dtype=np.uint32)
        filename = f"data/wikitext-103/{split}.bin"
        
        print(f"Saving {len(arr):,} tokens to {filename}...")
        memmap_arr = np.memmap(filename, dtype=np.uint32, mode='w+', shape=(len(arr),))
        memmap_arr[:] = arr[:]
        memmap_arr.flush()
        print("Done!")

if __name__ == "__main__":
    prepare_data()
