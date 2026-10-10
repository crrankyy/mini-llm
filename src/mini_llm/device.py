import torch

# check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"")

print(f"-" * 30)
print(f"Using device: {device}")
print(f"-" * 30)

