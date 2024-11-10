# src/train/metrics.py
import numpy as np
import torch
from typing import List, Dict
import logging
from pathlib import Path
import sys

# Add parent directory to path for importing eval_metrics
sys.path.append(str(Path(__file__).parent.parent.parent))
from eval_metrics import compute_piece_pitch_entropy, compute_piece_groove_similarity

class MusicMetrics:
    def __init__(self):
        self.reset()
        self.logger = logging.getLogger(__name__)

    def reset(self):
        """Reset metrics"""
        self.total_loss = 0
        self.num_batches = 0
        self.generated_samples = []
        
    def update_loss(self, loss: float):
        """Update running loss"""
        self.total_loss += loss
        self.num_batches += 1
        
    def get_average_loss(self) -> float:
        """Get average loss"""
        if self.num_batches == 0:
            return 0
        return self.total_loss / self.num_batches
    
    def compute_generation_metrics(self, 
                                 token_sequences: List[List[int]]) -> Dict[str, float]:
        """Compute H4 and GS metrics for generated sequences"""
        h4_scores = []
        gs_scores = []
        
        for tokens in token_sequences:
            try:
                # Compute metrics using eval_metrics functions
                h4 = compute_piece_pitch_entropy(tokens, window_size=4)
                gs = compute_piece_groove_similarity(tokens)
                
                h4_scores.append(h4)
                gs_scores.append(gs)
            except Exception as e:
                self.logger.warning(f"Error computing metrics: {str(e)}")
                continue
                
        if not h4_scores or not gs_scores:
            return {
                'h4_mean': 0.0,
                'h4_std': 0.0,
                'gs_mean': 0.0,
                'gs_std': 0.0
            }
            
        return {
            'h4_mean': np.mean(h4_scores),
            'h4_std': np.std(h4_scores),
            'gs_mean': np.mean(gs_scores),
            'gs_std': np.std(gs_scores)
        }

class TrainingMetrics(MusicMetrics):
    def __init__(self):
        super().__init__()
        self.reset_epoch()
        
    def reset_epoch(self):
        """Reset epoch-specific metrics"""
        self.epoch_loss = 0
        self.epoch_batches = 0
        
    def update_epoch_loss(self, loss: float):
        """Update epoch loss"""
        self.epoch_loss += loss
        self.epoch_batches += 1
        self.update_loss(loss)
        
    def get_epoch_metrics(self) -> Dict[str, float]:
        """Get metrics for current epoch"""
        if self.epoch_batches == 0:
            return {'loss': 0.0}
        
        return {
            'loss': self.epoch_loss / self.epoch_batches,
            'avg_loss': self.get_average_loss()
        }