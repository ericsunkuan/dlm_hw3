# src/model/model.py
import math
import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, List

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return x + self.pe[:x.size(0)]

class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        return self.pe[:, offset:offset + x.size(1)]

class MusicTransformerXL(nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 d_model: int = 512,
                 n_head: int = 8,
                 n_layers: int = 6,
                 d_ff: int = 2048,
                 dropout: float = 0.1,
                 mem_len: int = 512):
        super().__init__()
        
        self.d_model = d_model
        self.n_head = n_head
        self.mem_len = mem_len
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerXLLayer(
                d_model=d_model,
                n_head=n_head,
                d_ff=d_ff,
                dropout=dropout,
                mem_len=mem_len
            ) for _ in range(n_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Initialize parameters
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(self, x, mems=None):
        batch_size = x.size(0)
        
        # Initialize mems if None
        if mems is None:
            mems = [torch.empty(0, batch_size, self.d_model, device=x.device) 
                   for _ in range(len(self.transformer_layers))]
        
        # Get embeddings [batch, seq_len, d_model]
        hidden = self.token_embedding(x)
        hidden = self.dropout(hidden)
        
        # Transpose to [seq_len, batch, d_model] for attention
        hidden = hidden.transpose(0, 1)
        
        new_mems = []
        
        # Process through transformer layers
        for layer, mem in zip(self.transformer_layers, mems):
            hidden, new_mem = layer(hidden, mem)
            new_mems.append(new_mem.detach())  # Detach memory to prevent backprop through memory
        
        # Transpose back to [batch, seq_len, d_model]
        hidden = hidden.transpose(0, 1)
        
        # Output projection
        logits = self.output_layer(hidden)
        
        return logits, new_mems
    
    def generate(self, 
                input_ids: torch.Tensor, 
                max_length: int = 1024,
                temperature: float = 1.0,
                top_k: int = 50,
                top_p: float = 0.95):
        """Generate new tokens"""
        self.eval()
        current_seq = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids
        batch_size = current_seq.size(0)
        mems = None
        
        with torch.no_grad():
            for _ in range(max_length - current_seq.size(1)):
                # Get predictions
                logits, mems = self(current_seq, mems)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                current_seq = torch.cat([current_seq, next_token], dim=1)
        
        return current_seq

class TransformerXLLayer(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float, mem_len: int):
        super().__init__()
        
        self.mem_len = mem_len
        self.attention = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Feed forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
    
    def forward(self, x, memory=None):
        batch_size = x.size(1)
        
        # Handle memory
        if memory is not None and memory.size(0) > 0:
            # Ensure memory is on the correct device
            memory = memory.to(x.device)
            
            # Expand memory to match batch size if needed
            if memory.size(1) != batch_size:
                memory = memory.expand(-1, batch_size, -1)
            
            # Concatenate along sequence dimension
            x = torch.cat([memory, x], dim=0)
        
        # Self attention
        attended, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attended))
        
        # Feed forward
        x = self.norm2(x + self.dropout(self.ff(x)))
        
        # Return the full sequence and the memory for the next iteration
        return x, x[-self.mem_len:] if x.size(0) > self.mem_len else x

    def generate(self,
                prompt: torch.Tensor,
                max_length: int,
                temperature: float = 1.0,
                top_k: int = 0,
                top_p: float = 0.9,
                repetition_penalty: float = 1.2) -> torch.Tensor:
        """Generate sequence"""
        self.eval()
        current_seq = prompt.clone()
        
        # Reset memories
        self.memories = [None] * len(self.transformer_layers)
        
        with torch.no_grad():
            for _ in range(max_length - prompt.size(1)):
                # Get predictions
                logits, _ = self(current_seq)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for i in range(current_seq.size(1)):
                        next_token_logits[:, current_seq[:, i]] /= repetition_penalty
                
                # Filter logits
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                current_seq = torch.cat([current_seq, next_token], dim=1)
        
        return current_seq