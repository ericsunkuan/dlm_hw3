# src/train/train.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import yaml
from tqdm import tqdm
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from model.model import MusicTransformerXL
from data.dataset import MusicDataset
from data.preprocessing import MIDIPreprocessor
from metrics import TrainingMetrics

class Trainer:
    def __init__(self, config_path: str):
        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        # Setup logging
        self.setup_logging()
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.setup_components()
        
        # Initialize metrics
        self.metrics = TrainingMetrics()
        
    def setup_logging(self):
        """Setup logging and tensorboard"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = os.path.join('logs', timestamp)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Setup tensorboard
        self.writer = SummaryWriter(self.log_dir)
        
        # Setup file logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_components(self):
        """Initialize model, optimizer, and other components"""
        # Data preprocessing
        self.preprocessor = MIDIPreprocessor()
        
        # Dataset
        self.dataset = MusicDataset(
            data_dir=self.config['data']['data_dir'],
            preprocessor=self.preprocessor,
            max_seq_len=self.config['data']['sequence_length'],
            cache_dir='cache'
        )
        
        # Dataloader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            collate_fn=MusicDataset.collate_fn,
            pin_memory=True
        )
        
        # Model
        self.model = MusicTransformerXL(
            vocab_size=self.preprocessor.vocab_size,
            **self.config['model']
        ).to(self.device)
        
        # Optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate']
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training']['epochs']
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Gradient scaler for mixed precision training
        self.scaler = GradScaler()
        # Save dictionary for evaluation
        os.makedirs('cache', exist_ok=True)
        self.preprocessor.save_vocabulary('cache/dictionary.pkl')
        
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        self.metrics.reset_epoch()
        
        with tqdm(self.dataloader, desc=f'Epoch {epoch}') as pbar:
            for step, batch in enumerate(pbar):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass with mixed precision
                with autocast():
                    logits, _ = self.model(input_ids)
                    loss = self.criterion(logits.view(-1, self.preprocessor.vocab_size), 
                                       labels.view(-1))
                
                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Update metrics
                self.metrics.update_epoch_loss(loss.item())
                metrics = self.metrics.get_epoch_metrics()
                pbar.set_postfix(metrics)
                
                # Log to tensorboard
                global_step = epoch * len(self.dataloader) + step
                self.writer.add_scalar('train/loss', loss.item(), global_step)
                
                # Generate samples and compute metrics periodically
                if step % self.config['training']['eval_every_n_steps'] == 0:
                    self.evaluate_generation(epoch, step)
                
                # Save checkpoint
                if step % self.config['training']['save_every_n_steps'] == 0:
                    self.save_checkpoint(epoch, step)
                    
        return self.metrics.get_epoch_metrics()
    
    def evaluate_generation(self, epoch: int, step: int):
        """Generate samples and compute metrics"""
        self.model.eval()
        with torch.no_grad():
            # Generate a few samples
            prompt = torch.tensor([[
                self.preprocessor.special_tokens['BOS'],
                self.preprocessor.special_tokens['BAR'],
                self.preprocessor.vocab['Position_0']
            ]], device=self.device)
            
            generated_sequences = []
            for _ in range(5):  # Generate 5 samples
                generated = self.model.generate(
                    prompt=prompt,
                    max_length=512,  # Shorter for quicker evaluation
                    temperature=1.0,
                    top_k=40
                )
                generated_sequences.append(generated[0].cpu().tolist())
            
            # Compute metrics
            metrics = self.metrics.compute_generation_metrics(generated_sequences)
            
            # Log metrics
            for name, value in metrics.items():
                self.writer.add_scalar(f'generation/{name}', value, epoch * len(self.dataloader) + step)
        
        self.model.train()
    
    def save_checkpoint(self, epoch: int, step: int):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(self.log_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'metrics': self.metrics.get_epoch_metrics()
        }
        
        path = os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch}_step{step}.pt')
        torch.save(checkpoint, path)
        self.logger.info(f'Saved checkpoint: {path}')
        
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        
        for epoch in range(self.config['training']['epochs']):
            metrics = self.train_epoch(epoch)
            self.scheduler.step()
            
            # Log epoch metrics
            self.logger.info(f'Epoch {epoch} - Metrics: {metrics}')
            for name, value in metrics.items():
                self.writer.add_scalar(f'epoch/{name}', value, epoch)
            
            # Save checkpoint
            self.save_checkpoint(epoch, len(self.dataloader))
            
        self.logger.info("Training completed!")

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', type=str, required=True,
#                       help='Path to config file')
#     args = parser.parse_args()
    
#     trainer = Trainer(args.config)
#     trainer.train()

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the music model')
    
    parser.add_argument('--dict_path', 
                       type=str, 
                       required=True,
                       help='Path to dictionary file')
    
    parser.add_argument('--output_file_path', 
                       type=str, 
                       required=True,
                       help='Path to output directory')
    
    parser.add_argument('--config', 
                       type=str, 
                       required=True,
                       help='Path to config file')
    
    print("Before parsing arguments")
    print(f"Available arguments: {[action.dest for action in parser._actions]}")
    
    args = parser.parse_args()
    
    print("After parsing arguments")
    print(f"Parsed arguments: {vars(args)}")