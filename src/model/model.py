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
                top_k: int = 40) -> torch.Tensor:
        """
        Generate sequence from prompt
        Args:
            prompt: Starting sequence [batch_size, seq_len]
            max_length: Maximum length to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
        Returns:
            Generated sequence [batch_size, seq_len]
        """
        self.eval()
        current_seq = prompt.clone()
        batch_size = prompt.size(0)
        
        print(f"\nStarting generation with:")
        print(f"- Prompt shape: {prompt.shape}")
        print(f"- Max length: {max_length}")
        print(f"- Temperature: {temperature}")
        print(f"- Top-k: {top_k}")
        
        with torch.no_grad():
            for i in range(max_length - prompt.size(1)):
                if i % 50 == 0:
                    print(f"Generating token {i}/{max_length - prompt.size(1)}")
                
                # Get model predictions for the last token
                outputs = self(current_seq)
                # Handle tuple output (logits, mems)
                if isinstance(outputs, tuple):
                    logits = outputs[0]  # Get just the logits
                else:
                    logits = outputs
                
                # Get the last token predictions
                logits = logits[:, -1, :]  # [batch_size, vocab_size]
                
                # Print shape information for debugging
                # print(f"Logits shape: {logits.shape}")
                
                # Apply temperature
                logits = logits / temperature
                
                # Apply top-k sampling
                if top_k > 0:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                # Convert to probabilities
                probs = torch.softmax(logits, dim=-1)
                
                try:
                    # Sample next token
                    next_token = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]
                    
                    # Append to sequence
                    current_seq = torch.cat([current_seq, next_token], dim=1)
                    
                    # Optional: check for end token
                    if hasattr(self, 'eos_token_id') and (next_token == self.eos_token_id).any():
                        break
                        
                except Exception as e:
                    print(f"Error during token generation: {str(e)}")
                    print(f"Probabilities shape: {probs.shape}")
                    print(f"Probabilities sum: {probs.sum().item()}")
                    print(f"Any NaN in probabilities: {torch.isnan(probs).any().item()}")
                    print(f"Any inf in logits: {torch.isinf(logits).any().item()}")
                    raise e
                
                # Clear GPU cache periodically
                if i % 100 == 0:
                    torch.cuda.empty_cache()
        
        print("Generation complete!")
        return current_seq