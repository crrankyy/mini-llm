import torch
from torch.utils.data import Dataset, DataLoader, Sampler
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
    
def create_dataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    dataset = CustomDataset(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader

import numpy as np

class MemMapDataset(Dataset):
    def __init__(self, bin_file_path, max_length, stride):
        self.data = np.memmap(bin_file_path, dtype=np.uint32, mode='r')
        self.max_length = max_length
        self.stride = stride
        self.num_samples = (len(self.data) - self.max_length) // self.stride

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        chunk = self.data[start_idx : start_idx + self.max_length + 1].astype(np.int64)
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

class ResumableSampler(Sampler):
    def __init__(self, num_samples, seed):
        self.num_samples = num_samples
        self.seed = seed
        self.epoch = 0
        self.start = 0

    def set_position(self, epoch, start):
        self.epoch = epoch
        self.start = start

    def __iter__(self):
        # the order depends only on the seed and the epoch, so a later run can
        # rebuild it and carry on from the sample where an earlier run stopped
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.num_samples, generator=generator)
        return iter(order[self.start:].tolist())

    def __len__(self):
        return max(self.num_samples - self.start, 0)

def create_memmap_dataloader(bin_file_path, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0, seed=None):
    dataset = MemMapDataset(bin_file_path, max_length, stride)
    if seed is not None:
        sampler = ResumableSampler(len(dataset), seed)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=drop_last, num_workers=num_workers)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader