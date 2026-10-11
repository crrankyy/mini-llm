import torch
import torch.nn as nn
import tiktoken

from mini_llm.model.layers import LayerNorm, GELU, FeedForward
from mini_llm.model.block import TransformerBlock

GPT_CONFIG_124M = {
    "vocab_size": 100277, 
    "context_length": 1024, 
    "embedding_dimension": 768, 
    "n_heads": 12, 
    "n_layers": 12, 
    "drop_rate": 0.1, 
    "qkv_bias": False 
}

class DummyGPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(
                config["vocab_size"], config["embedding_dimension"]
        )
        self.position_embedding = nn.Embedding(
                config["context_length"], config["embedding_dimension"]
        )
        
        self.drop_embedding = nn.Dropout( config["drop_rate"] )

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(config)
                for _ in range(config["n_layers"])
            ]
        )

        self.final_norm = LayerNorm( config["embedding_dimension"] )

        self.out_head = nn.Linear(
                config["embedding_dimension"], config["vocab_size"], bias = False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        
        token_embedding = self.token_embedding(in_idx)
        
        position_embedding = self.position_embedding(
            torch.arange(seq_len, device=in_idx.device)
        )
        
        x = token_embedding + position_embedding
        
        x = self.drop_embedding(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        
        logits = self.out_head(x)
        
        return logits
        
    

class ExampleDeepNeuralNetwork(nn.Module):
    
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([ #1
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]),
            GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]),
            GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]),
            GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]),
            GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]),
            GELU())
        ])
    
    def forward(self, x):
        
        for layer in self.layers:
            layer_output = layer(x) #2
            if self.use_shortcut and x.shape == layer_output.shape: #3
                x = x + layer_output
            else:
                x = layer_output
        
        return x
    

def print_gradients(model, x):
    output = model(x) #1
    target = torch.tensor([[0.]])
    loss = nn.MSELoss()
    loss = loss(output, target) #2
    loss.backward() #3
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")


tokenizer = tiktoken.get_encoding("cl100k_base")

vocab_size = tokenizer.n_vocab

print("Vocab size: ", vocab_size)

batch = []

txt1 = "Every effort moves you"
txt2 = "Every day holds surprise"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))

batch = torch.stack(batch, dim=0)

torch.manual_seed(123)

model = DummyGPT(GPT_CONFIG_124M)
logits = model(batch)
print("Output shape: ", logits.shape)
print(logits)

mean = logits.mean(dim=-1, keepdim=True)
var  = logits.var(dim=-1, keepdim=True)
print("Normalized Mean:\n", mean)
print("Normalized Variance:\n", var)

print("-"*100)
print("-"*100)

batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print(out)
print("")

mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)

print("-"*100)

out_norm = (out - mean) / torch.sqrt(var)

mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
print("Normalized Mean:\n", mean)
print("Normalized Variance:\n", var)

print("-"*100)
print("-"*100)

ln = LayerNorm(embedding_dimension=5)
out_ln = ln(batch_example)

mean = out_ln.mean(dim=-1, keepdim=True)
var  = out_ln.var(dim=-1, keepdim=True)
print("Normalized Mean:\n", mean)
print("Normalized Variance:\n", var)

print("-"*100)
print("-"*100)

ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)
out = ffn(x)
print(out.shape)

print("-"*100)
print("-"*100)

layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1., 0., -1.]])

model_without_shortcut = ExampleDeepNeuralNetwork(
                                layer_sizes, use_shortcut=False
                        )

print("Gradients of Model without Shortcuts: ")
print_gradients(model_without_shortcut, sample_input)

print("-"*100)
print("-"*100)

model_with_shortcut = ExampleDeepNeuralNetwork(
                            layer_sizes, use_shortcut=True
                    )

print("Gradients of Model with Shortcuts: ")
print_gradients(model_with_shortcut, sample_input)

print("-"*100)
print("-"*100)

x = torch.rand(2, 4, 768)
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)
print("Input shape:", x.shape)
print("Output shape:", output.shape)