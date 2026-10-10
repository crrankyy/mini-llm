import torch
import torch.nn as nn
import tiktoken

from mini_llm.generation import textgen_v1
from mini_llm.model.gpt import GPTModel
from mini_llm.generation import text_to_token_ids, token_ids_to_text
from mini_llm.data import create_dataloader
from mini_llm.training.loss import calc_loss_loader
from mini_llm.training.trainer import train_model_v1

from mini_llm.config import GPT_CONFIG_124M
    

device = torch.device("cpu")
print(f"")
print(f"-" * 100)
print(f"Using device: {device}")
print(f"-" * 100)


tokenizer = tiktoken.get_encoding("cl100k_base")

batch = []

txt1 = "Every effort moves you"
txt2 = "Every day holds surprise"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))

batch = torch.stack(batch, dim=0)



torch.manual_seed(123)

model = GPTModel(GPT_CONFIG_124M)

print("-" * 100)
print("-" * 100)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")

total_size_bytes = total_params * 4
total_size_mb = total_size_bytes / (1024 * 1024)
print(f"Total size of the model: {total_size_mb:.2f} MB")

print("-" * 100)
print("-" * 100)

start_context = "Every effort moves you"
print("Start Context: ", start_context)

print("-" * 100)

model.eval()

token_ids = textgen_v1(
                model = model,
                idx = text_to_token_ids(start_context, tokenizer),
                max_new_tokens = 10,
                context_size = GPT_CONFIG_124M["context_length"]
)

print("Output:", token_ids_to_text(token_ids=token_ids, tokenizer=tokenizer))

print("-" * 100)
print("-" * 100)

with open("data/basic-data.txt", "r") as f:
    raw_text = f.read()

encoded_text = tokenizer.encode(raw_text)

total_characters = len(raw_text)
total_tokens = len(encoded_text)

print("Characters: ", total_characters)
print("Tokens: ", total_tokens)
print("")

train_ratio = 0.90
split_idx = int(train_ratio * total_characters)

train_data = raw_text[:split_idx]
val_data = raw_text[split_idx:]

train_dataloader = create_dataloader(
                        train_data,
                        batch_size=2,
                        max_length=GPT_CONFIG_124M["context_length"],
                        stride=GPT_CONFIG_124M["context_length"],
                        shuffle=True,
                        drop_last=True,
                        num_workers=0
                )

val_dataloader   = create_dataloader(
                        val_data,
                        batch_size=2,
                        max_length=GPT_CONFIG_124M["context_length"],
                        stride=GPT_CONFIG_124M["context_length"],
                        shuffle=False,
                        drop_last=False,
                        num_workers=0
                )

print("Train loader:")

for x, y in train_dataloader:
    print(x.shape, y.shape)


print("\nValidation loader:")

for x, y in val_dataloader:
    print(x.shape, y.shape)

model.to(device)

with torch.no_grad(): 
    train_loss = calc_loss_loader(train_dataloader, model, device)
    val_loss = calc_loss_loader(val_dataloader, model, device)
print("Training loss:", train_loss)
print("Validation loss:", val_loss)

print("-" * 100)
print("-" * 100)


model = GPTModel(GPT_CONFIG_124M)

model.to(device)
optimizer = torch.optim.AdamW(
                model.parameters(), #1
                lr=0.0004, weight_decay=0.1
            )

num_epochs = 10

train_losses, val_losses, tokens_seen = train_model_v1(
    model, train_dataloader, val_dataloader, optimizer, device,
    num_epochs=num_epochs, eval_frq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)