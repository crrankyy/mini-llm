import argparse
import torch
import torch.nn as nn
import tiktoken

from mini_llm.generation import textgen_v1
from mini_llm.model.gpt import GPTModel
from mini_llm.generation import text_to_token_ids, token_ids_to_text
from mini_llm.data import create_memmap_dataloader
from mini_llm.training.loss import calc_loss_loader
from mini_llm.training.trainer import train_model_v1

from mini_llm.config import GPT_CONFIG_124M
from mini_llm.device import get_device

parser = argparse.ArgumentParser(description="Train the GPT model on WikiText-103.")
parser.add_argument(
    "--checkpoint-dir", default="checkpoints",
    help="where checkpoints and the final weights are written, and where a resume looks for them"
)
checkpoint_dir = parser.parse_args().checkpoint_dir

# Use GPU if available
device = get_device()
print(f"")
print(f"-" * 100)
print(f"Using device: {device}")
print(f"-" * 100)

tokenizer = tiktoken.get_encoding("cl100k_base")

torch.manual_seed(123)

model = GPTModel(GPT_CONFIG_124M)
model.to(device)

print("-" * 100)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")
total_size_bytes = total_params * 4
total_size_mb = total_size_bytes / (1024 * 1024)
print(f"Total size of the model: {total_size_mb:.2f} MB")
print("-" * 100)

import os

data_dir = "data/wikitext-103"
if os.path.exists("/content"):
    local_data_dir = "/content/data/wikitext-103"
    if not os.path.exists(local_data_dir):
        print("Moving data to local Colab SSD for faster reading (this takes a minute)...")
        import shutil
        shutil.copytree(data_dir, local_data_dir)
    data_dir = local_data_dir

print("Loading data via memmap...")
train_dataloader = create_memmap_dataloader(
    bin_file_path=f"{data_dir}/train.bin",
    batch_size=16, # Increased for T4 GPU
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    shuffle=True,
    drop_last=True,
    num_workers=2
)

val_dataloader = create_memmap_dataloader(
    bin_file_path=f"{data_dir}/validation.bin",
    batch_size=16,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    shuffle=False,
    drop_last=False,
    num_workers=2
)

print("Train loader initialized. Number of batches:", len(train_dataloader))
print("Val loader initialized. Number of batches:", len(val_dataloader))

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-4, weight_decay=0.1
)

num_epochs = 5 # Total number of epochs you want to train for

import glob
start_epoch = 0
checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "gpt_wikitext_checkpoint_ep*.pth")))
if checkpoints:
    latest_checkpoint = checkpoints[-1]
    print(f"Found checkpoint: {latest_checkpoint}. Resuming training...")
    checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
    
    # Handle the fact that torch.compile adds an _orig_mod. prefix to keys
    state_dict = checkpoint['model_state_dict']
    unwrapped_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(unwrapped_state_dict)
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']

# compiled only after the resume: the wrapper torch.compile returns expects the
# _orig_mod. prefix on every key, so a checkpoint has to be loaded into the plain model.
# skipped on MPS, where torch.compile returns without error and then fails at the first forward pass
if device.type != "mps":
    print("Compiling the model using PyTorch 2.0 (this takes a minute but speeds up training)...")
    try:
        model = torch.compile(model)
    except Exception as e:
        print(f"torch.compile not supported or failed: {e}. Running in eager mode.")

if start_epoch >= num_epochs:
    print(f"Already completed {num_epochs} epochs. Training is done!")
else:
    print(f"Starting training from epoch {start_epoch + 1}...")
    try:
        train_losses, val_losses, tokens_seen = train_model_v1(
            model, train_dataloader, val_dataloader, optimizer, device,
            num_epochs=num_epochs - start_epoch, eval_frq=500, eval_iter=10,
            start_context="The secret to training large models is", tokenizer=tokenizer, start_epoch=start_epoch,
            checkpoint_dir=checkpoint_dir
        )
    except KeyboardInterrupt:
        print("\n[!] Training manually stopped! Saving emergency checkpoint...")
        interrupted_path = os.path.join(checkpoint_dir, "gpt_wikitext_checkpoint_interrupted.pth")
        torch.save({
            'epoch': start_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, interrupted_path)
        print(f"--> Emergency checkpoint saved to '{interrupted_path}'!")
        print("You can safely close Colab now. Next time you run this, it will resume from here.")

print("Training script finished.")

print("Training complete. Saving model...")
final_path = os.path.join(checkpoint_dir, "gpt_wikitext_124M.pth")
torch.save(model.state_dict(), final_path)
print(f"Model saved to {final_path}!")
