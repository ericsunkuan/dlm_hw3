# src/train/train2.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast
import yaml
from tqdm import tqdm
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from model.model import MusicTransformerXL
from data.dataset import MusicDataset
from data.preprocessing import MIDITokenizer
from metrics import TrainingMetrics
from transformers import get_cosine_schedule_with_warmup

class EnhancedTrainer:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Initialize curriculum stages
        self.curriculum_stages = [
            {'max_seq_len': 256, 'epochs': 2, 'temperature': 1.0},
            {'max_seq_len': 512, 'epochs': 2, 'temperature': 0.9}
        ]
        
        # Initialize other attributes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_logging()
        self.setup_components()
        self.metrics = TrainingMetrics()
        
        # Add gradient accumulation
        self.gradient_accumulation_steps = self.config['training'].get('gradient_accumulation_steps', 2)
        
        # Add memory management
        torch.backends.cudnn.benchmark = True
        self.empty_cache_freq = 100
    
    def setup_logging(self):
        """Enhanced logging setup"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = os.path.join('logs', timestamp)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Setup tensorboard with more categories
        self.writer = SummaryWriter(self.log_dir)
        
        # Create error log handler separately
        error_handler = logging.FileHandler(os.path.join(self.log_dir, 'errors.log'))
        error_handler.setLevel(logging.ERROR)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, 'training.log')),
                logging.StreamHandler(),
                error_handler  # Add the error handler here
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_components(self):
        """Enhanced component initialization"""
        # Initialize tokenizer
        self.tokenizer = MIDITokenizer()
        
        # Get training config with defaults
        train_config = self.config.get('training', {})
        batch_size = train_config.get('batch_size', 16)
        num_workers = train_config.get('num_workers', 4)
        pin_memory = train_config.get('pin_memory', True)
        val_batch_size = train_config.get('val_batch_size', batch_size)
        
        # Dataset with validation split
        data_dir = self.config['data']['data_dir']
        self.logger.info(f"Loading dataset from {data_dir}")
        
        try:
            self.dataset = MusicDataset(
                data_dir=data_dir,
                tokenizer=self.tokenizer,
                max_seq_len=self.config['data']['max_seq_len'],
                overlap=self.config['data']['overlap']
            )
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {str(e)}")
            raise
        
        self.logger.info(f"Dataset loaded with {len(self.dataset)} sequences")
        
        if len(self.dataset) == 0:
            raise ValueError("Dataset is empty")
        
        # Create train/val split
        train_size = int(self.config['data'].get('train_val_split', 0.9) * len(self.dataset))
        val_size = len(self.dataset) - train_size
        
        if train_size == 0 or val_size == 0:
            raise ValueError(f"Invalid split sizes: train_size={train_size}, val_size={val_size}")
            
        train_dataset, val_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size]
        )
        
        # Create dataloaders
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )
        
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers // 2,  # Use fewer workers for validation
            pin_memory=pin_memory
        )
        
        # Create model
        model_config = self.config.get('model', {})
        self.model = MusicTransformerXL(
            vocab_size=self.tokenizer.vocab_size,
            d_model=model_config.get('d_model', 512),
            n_head=model_config.get('n_head', 8),
            n_layers=model_config.get('n_layers', 6),
            d_ff=model_config.get('d_ff', 2048),
            dropout=model_config.get('dropout', 0.1),
            mem_len=model_config.get('mem_len', 512)
        ).to(self.device)
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_config.get('learning_rate', 1e-4),
            weight_decay=train_config.get('weight_decay', 0.01)
        )
        
        # Setup scheduler
        total_steps = len(self.train_dataloader) * sum(stage['epochs'] for stage in self.curriculum_stages)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=train_config.get('warmup_steps', 100),
            num_training_steps=total_steps
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Gradient scaler for mixed precision training
        self.scaler = GradScaler()
        
        # Save dictionary
        os.makedirs('cache', exist_ok=True)
        self.tokenizer.save_vocabulary('cache/dictionary.pkl')
        
        # Save tokenizer vocabulary
        vocab_path = 'cache/vocabulary.json'  # Changed extension and path
        try:
            self.tokenizer.save_vocabulary(vocab_path)
        except Exception as e:
            self.logger.warning(f"Could not save vocabulary: {e}")
            self.logger.warning(f"Continuing without saving vocabulary")
        
    def train_epoch(self, epoch: int, stage: dict):
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(tqdm(self.train_dataloader)):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            with torch.cuda.amp.autocast():
                logits, _ = self.model(input_ids)
                loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                
            # Backward pass
            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
            
            total_loss += loss.item()
            
            # Log metrics
            if batch_idx % 100 == 0:
                self.logger.info(f'Epoch {epoch} | Batch {batch_idx} | Loss {loss.item():.4f}')
                
            # Clear GPU cache periodically
            if batch_idx % self.empty_cache_freq == 0:
                torch.cuda.empty_cache()
        
        return {'loss': total_loss / len(self.train_dataloader)}
    
    @torch.amp.autocast('cuda')
    def evaluate(self, epoch: int, step: int):
        """Optimized validation step"""
        self.model.eval()
        val_metrics = {}
        val_loss = 0
        num_batches = min(len(self.val_dataloader), 50)  # Limit validation batches
        
        with torch.no_grad():
            for i, batch in enumerate(self.val_dataloader):
                if i >= num_batches:
                    break
                    
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                
                logits, _ = self.model(input_ids)
                loss = self.criterion(
                    logits.view(-1, self.tokenizer.vocab_size),
                    labels.view(-1)
                )
                
                val_loss += loss.item()
            
            val_loss /= num_batches
            val_metrics['val_loss'] = val_loss
            
            # Log validation metrics less frequently
            if step % 20 == 0:
                global_step = epoch * len(self.train_dataloader) + step
                self.writer.add_scalar('validation/loss', val_loss, global_step)
            
        return val_metrics
    
    def train(self):
        """Enhanced training loop with curriculum learning"""
        self.logger.info("Starting training with curriculum learning...")
        best_val_loss = float('inf')
        
        for stage_idx, stage in enumerate(self.curriculum_stages):
            self.logger.info(f"Starting curriculum stage {stage_idx + 1}")
            self.logger.info(f"Max sequence length: {stage['max_seq_len']}")
            self.logger.info(f"Temperature: {stage['temperature']}")
            
            for epoch in range(stage['epochs']):
                # Training
                train_metrics = self.train_epoch(epoch, stage)
                
                # Validation
                val_metrics = self.evaluate(epoch, len(self.train_dataloader))
                
                # Log combined metrics
                self.logger.info(f'Stage {stage_idx + 1}, Epoch {epoch}')
                self.logger.info(f'Train metrics: {train_metrics}')
                self.logger.info(f'Validation metrics: {val_metrics}')
                
                # Save best model
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    self.save_checkpoint(epoch, len(self.train_dataloader), 
                                      train_metrics, val_metrics, is_best=True)
        
        self.logger.info("Training completed!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train the music model')
    parser.add_argument('--config_path', type=str, default='configs/model_config.yaml',
                       help='Path to config file')
    parser.add_argument('--output_file_path', type=str, required=True,
                       help='Path to output directory')
    parser.add_argument('--max_files', type=int, default=500,
                       help='Maximum number of MIDI files to process')
    
    args = parser.parse_args()
    
    trainer = EnhancedTrainer(
        config_path=args.config_path
    )
    trainer.output_path = args.output_file_path
    
    trainer.train()