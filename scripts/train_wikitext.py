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
from mini_llm.training.checkpoint import load_latest_checkpoint, training_progress

from mini_llm.config import GPT_CONFIG_124M
from mini_llm.device import get_device

parser = argparse.ArgumentParser(description="Train the GPT model on prepared token files.")
parser.add_argument(
    "--data-dir", default="data/wikitext-103",
    help="folder holding train.bin and validation.bin, as the prepare scripts write them"
)
parser.add_argument(
    "--epochs", type=int, default=5,
    help="passes over the training data in total, counting the ones earlier runs completed"
)
parser.add_argument(
    "--checkpoint-dir", default="checkpoints",
    help="where checkpoints and the final weights are written, and where a resume looks for them"
)
parser.add_argument(
    "--save-every", type=int, default=2000,
    help="training steps between saves of the rolling checkpoint, which a later run resumes from"
)
parser.add_argument(
    "--keep-checkpoints", type=int, default=10,
    help="how many end-of-epoch checkpoints to keep; older ones are deleted"
)
args = parser.parse_args()
if args.save_every < 1 or args.keep_checkpoints < 1 or args.epochs < 1:
    parser.error("--save-every, --keep-checkpoints and --epochs must be at least 1")
checkpoint_dir = args.checkpoint_dir

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

data_dir = args.data_dir
# data kept on Google Drive is copied to the Colab machine's own disk first;
# data that is already on that disk is read where it is
if os.path.realpath(data_dir).startswith("/content/drive/"):
    local_data_dir = os.path.join("/content/data", os.path.basename(os.path.normpath(data_dir)))
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
    drop_last=True,
    num_workers=2,
    # a fixed order per epoch, so a run that stops inside an epoch can be continued where it stopped
    seed=123
)

# evaluation keeps drawing random training batches, which the ordered loader above cannot give
train_eval_dataloader = create_memmap_dataloader(
    bin_file_path=f"{data_dir}/train.bin",
    batch_size=16,
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

num_epochs = args.epochs # Total number of epochs you want to train for

start_epoch = 0
resume = None
checkpoint, latest_checkpoint = load_latest_checkpoint(checkpoint_dir)
if checkpoint is not None:
    print(f"Found checkpoint: {latest_checkpoint}. Resuming training...")
    
    # Handle the fact that torch.compile adds an _orig_mod. prefix to keys
    state_dict = checkpoint['model_state_dict']
    unwrapped_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(unwrapped_state_dict)
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    resume = training_progress(checkpoint, latest_checkpoint)

    # the copy that was read from disk is no longer needed and is as large as the file
    del checkpoint, state_dict, unwrapped_state_dict

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
    batches_done = resume["batches_done"] if resume is not None else 0
    print(f"Starting training from epoch {start_epoch + 1}, after {batches_done} of its batches...")
    try:
        train_losses, val_losses, tokens_seen = train_model_v1(
            model, train_dataloader, val_dataloader, optimizer, device,
            num_epochs=num_epochs - start_epoch, eval_frq=500, eval_iter=10,
            start_context="The secret to training large models is", tokenizer=tokenizer, start_epoch=start_epoch,
            checkpoint_dir=checkpoint_dir, save_frq=args.save_every, keep_checkpoints=args.keep_checkpoints,
            resume=resume, eval_train_loader=train_eval_dataloader
        )
    except KeyboardInterrupt:
        # the trainer has already saved the position it had reached
        print("\n[!] Training manually stopped! The next run resumes from the rolling checkpoint.")

print("Training script finished.")

print("Training complete. Saving model...")
final_path = os.path.join(checkpoint_dir, "gpt_wikitext_124M.pth")
torch.save(model.state_dict(), final_path)
print(f"Model saved to {final_path}!")
