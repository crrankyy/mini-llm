import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class CustomDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length + 1, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
def create_dataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    dataset = CustomDataset(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def main():

    with open("data/basic-data.txt", "r") as f:
        raw_text = f.read()

    dataloader = create_dataloader(raw_text, batch_size=1, max_length=8, stride=4, shuffle=False)

    data_iter = iter(dataloader)

    print("-" * 100)
    first_batch = next(data_iter)
    print("Input IDs:", first_batch[0])
    print("Target IDs:", first_batch[1])
    print("-" * 100)
    second_batch = next(data_iter)
    print("Input IDs:", second_batch[0])
    print("Target IDs:", second_batch[1])
    print("-" * 100)

    dataloader = create_dataloader(raw_text, batch_size=4, max_length=8, stride=4, shuffle=False)

    data_iter = iter(dataloader)

    print("-" * 100)
    first_batch = next(data_iter)
    print("Input IDs:", first_batch[0])
    print("Target IDs:", first_batch[1])
    print("-" * 100)
    second_batch = next(data_iter)
    print("Input IDs:", second_batch[0])
    print("Target IDs:", second_batch[1])
    print("-" * 100)


if __name__ == "__main__":
    main()
