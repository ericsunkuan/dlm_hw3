# src/model/model.py
import math
import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple

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

class MusicTransformerXL(nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 d_model: int = 512,
                 n_head: int = 8,
                 n_layers: int = 12,
                 d_ff: int = 2048,
                 dropout: float = 0.1,
                 mem_len: int = 512):
        super().__init__()
        
        self.d_model = d_model
        self.mem_len = mem_len
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # TransformerXL layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Initialize memory
        self.mems = []
        
    def init_memory(self, batch_size: int, device: torch.device):
        """Initialize memory for sequence generation"""
        self.mems = [torch.zeros(batch_size, self.mem_len, self.d_model, device=device)]
        
    def _update_memory(self, hidden: torch.Tensor, mems: torch.Tensor) -> torch.Tensor:
        """Update memory with current hidden states"""
        if mems is None:
            return hidden
        current_length = hidden.size(1)
        mem_length = mems.size(1)
        with torch.no_grad():
            new_memory = torch.cat([mems, hidden], dim=1)[:, -self.mem_len:]
        return new_memory
        
    def forward(self, 
                src: torch.Tensor,
                memory: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len]
            memory: Optional tensor for cached memory
            src_mask: Optional mask for padding
        """
        # Embedding and positional encoding
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        
        # Concatenate with memory if provided
        if memory is not None:
            x = torch.cat([memory, x], dim=1)
            
        # Transform
        output = self.transformer(x, src_key_padding_mask=src_mask)
        
        # Update memory
        new_memory = self._update_memory(output, memory)
        
        # Get predictions
        logits = self.output_layer(output)
        
        return logits, new_memory
    
    def generate(self, 
                prompt: torch.Tensor,
                max_length: int,
                temperature: float = 1.0,
                top_k: int = 0,
                top_p: float = 0.9) -> torch.Tensor:
        """Generate sequence from prompt"""
        self.eval()
        current_seq = prompt.clone()
        batch_size = prompt.size(0)
        memory = None
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get predictions
                logits, memory = self(current_seq, memory)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample from the filtered distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                current_seq = torch.cat([current_seq, next_token], dim=1)
                
                # Break if EOS token is generated
                if (next_token == 2).any():  # Assuming 2 is the EOS token
                    break
                    
        return current_seq