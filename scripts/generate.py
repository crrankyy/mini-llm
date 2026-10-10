import argparse
import glob
import torch
import tiktoken
import sys
import os

from mini_llm.model.gpt import GPTModel
from mini_llm.config import GPT_CONFIG_124M
from mini_llm.device import get_device

def parse_args():
    parser = argparse.ArgumentParser(description="Generate text from trained weights at an interactive prompt.")
    parser.add_argument(
        "--weights", default=None,
        help="weights or checkpoint file to load; skips the search in --checkpoint-dir"
    )
    parser.add_argument(
        "--checkpoint-dir", default="checkpoints",
        help="where to look for the final weights or, failing that, the latest checkpoint"
    )
    return parser.parse_args()

def find_weights(checkpoint_dir):
    final_path = os.path.join(checkpoint_dir, "gpt_wikitext_124M.pth")
    if os.path.exists(final_path):
        return final_path

    # Fallback to check if they have a checkpoint
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "gpt_wikitext_checkpoint_*.pth")))
    if checkpoints:
        return checkpoints[-1]

    return final_path

def load_model_weights(model, path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No weight file found at: {path}")
        
    print(f"Loading weights from {path}...")
    state_dict = torch.load(path, map_location=device, weights_only=False)
    
    # Extract state dict if saved inside a checkpoint dictionary
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
        
    # Strip '_orig_mod.' prefix added by torch.compile
    clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    print("Weights loaded successfully!")
    return model

def generate_text(model, tokenizer, prompt, device, max_new_tokens=100, temperature=0.7, top_k=50):
    model.eval()
    context_size = GPT_CONFIG_124M["context_length"]
    
    # Get end of text token ID
    try:
        eot_token_id = tokenizer.encode("<|endoftext|>", allowed_special="all")[0]
    except Exception:
        eot_token_id = 100257
        
    # Encode prompt
    encoded = tokenizer.encode(prompt, allowed_special="all")
    idx = torch.tensor(encoded).unsqueeze(0).to(device)
    
    # Generate tokens one by one
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
            
        logits = logits[:, -1, :]
        
        # Apply top-k filtering
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val.unsqueeze(-1), torch.tensor(float('-inf')).to(logits.device), logits)
            
        # Apply temperature scaling and sample
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            
        idx = torch.cat((idx, idx_next), dim=1)
        
        # Stop if we hit the end of text token
        if idx_next.item() == eot_token_id:
            break
            
    # Decode and return only the newly generated text
    generated_tokens = idx[0, len(encoded):].tolist()
    return tokenizer.decode(generated_tokens)

def run_cli():
    args = parse_args()

    device = get_device()
    print(f"Using device: {device}")
    
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    # Initialize model architecture
    model = GPTModel(GPT_CONFIG_124M)
    
    # Load weights
    weight_path = args.weights or find_weights(args.checkpoint_dir)

    try:
        model = load_model_weights(model, weight_path, device)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Put the weights or a checkpoint in '{args.checkpoint_dir}', or pass a file with --weights.")
        return
        
    model.to(device)
    
    print("\n" + "="*50)
    print("Interactive Chat/Generation Mode")
    print("Type your prompt and press Enter. Type 'quit' or 'exit' to stop.")
    print("="*50 + "\n")
    
    while True:
        try:
            prompt = input("Prompt >>> ")
            if prompt.strip().lower() in ['quit', 'exit']:
                break
            if not prompt.strip():
                continue
                
            print("\nResponse:", end=" ", flush=True)
            response = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_new_tokens=150,
                temperature=0.8,
                top_k=40
            )
            print(response.strip() + "\n")
            print("-" * 50 + "\n")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    run_cli()
