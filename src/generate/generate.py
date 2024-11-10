# src/generate.py
import os
import torch
import yaml
import logging
from typing import List, Optional, Dict
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast
from miditoolkit import MidiFile
import subprocess
from datetime import datetime
import miditoolkit
# Add parent directory to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from model.model import MusicTransformerXL
from data.preprocessing import MIDIPreprocessor
from train.setup import create_directory_structure

class MusicGenerator:
    def __init__(self, 
                 checkpoint_path: str,
                 config_path: str,
                 output_dir: str = 'outputs'):
        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging first
        self.setup_logging()
        
        # Create output directories
        self.output_dir = create_directory_structure(output_dir)
        self.logger.info(f"Created output directories in: {output_dir}")
        
        # Setup components
        self.setup_components(checkpoint_path)
        
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Logging initialized")
        
    def setup_components(self, checkpoint_path: str):
        """Initialize model and preprocessor"""
        self.logger.info(f"Loading checkpoint from: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.logger.info("Checkpoint loaded successfully")
        
        # Load event dictionary
        import pickle
        dict_path = "/home/paperspace/dlmusic/hw3/tutorial/basic_event_dictionary.pkl"
        with open(dict_path, 'rb') as f:
            self.event2word, self.word2event = pickle.load(f)
        self.logger.info(f"Loaded dictionary with {len(self.event2word)} events")
        
        # Initialize preprocessor
        self.logger.info("Initializing preprocessor...")
        self.preprocessor = MIDIPreprocessor()
        self.logger.info(f"Preprocessor vocab size: {self.preprocessor.vocab_size}")
        
        # Initialize model
        self.logger.info("Initializing model...")
        model_config = self.config['model']
        self.logger.info(f"Model config: {model_config}")
        
        self.model = MusicTransformerXL(
            vocab_size=self.preprocessor.vocab_size,
            **model_config
        ).to(self.device)
        self.logger.info(f"Model created on device: {self.device}")
        
        # Load model weights
        self.logger.info("Loading model weights...")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.logger.info("Model weights loaded successfully")
        
    def tokens_to_midi(self, tokens: List[int], output_path: str):
        """Convert token sequence to MIDI file with correct event formatting"""
        self.logger.info(f"Converting tokens to MIDI: {output_path}")
        
        # Create MIDI file
        midi = miditoolkit.midi.parser.MidiFile()
        midi.ticks_per_beat = 480  # Standard MIDI resolution
        
        # Create instrument track (piano by default)
        instrument = miditoolkit.midi.containers.Instrument(
            program=0,  # Piano
            is_drum=False,
            name='Piano'
        )
        
        # Initialize state variables
        current_time = 0
        current_bar = 0
        current_position = 0
        current_tempo = 120  # Default tempo
        current_velocity = 64  # Default velocity
        notes_to_add = []
        
        # Convert tokens back to events
        events = [self.word2event[token] for token in tokens]
        self.logger.info(f"First few events: {events[:10]}")
        
        # Process events
        current_note = None
        for event in events:
            try:
                # Bar marker
                if event == 'Bar_None':
                    current_bar += 1
                    current_position = 0
                    continue
                    
                # Position
                if event.startswith('Position_'):
                    position_str = event.split('_')[1].replace('/16', '')
                    current_position = (int(position_str) - 1) / 16
                    continue
                    
                # Note On
                if event.startswith('Note On_'):
                    if current_note is not None:
                        # Add previous note if exists
                        notes_to_add.append(current_note)
                    
                    pitch = int(event.split('_')[1])
                    current_time = int((current_bar * 4 + current_position) * midi.ticks_per_beat)  # Convert to int
                    current_note = miditoolkit.midi.containers.Note(
                        velocity=current_velocity,
                        pitch=pitch,
                        start=current_time,
                        end=current_time  # Will be updated when duration is set
                    )
                    
                # Note Velocity
                if event.startswith('Note Velocity_'):
                    current_velocity = int(event.split('_')[1])
                    if current_note:
                        current_note.velocity = current_velocity
                        
                # Note Duration
                if event.startswith('Note Duration_'):
                    if current_note:
                        duration_index = int(event.split('_')[1])
                        duration_ticks = int((duration_index + 1) * 120)  # Convert to int
                        current_note.end = current_time + duration_ticks
                        notes_to_add.append(current_note)
                        current_note = None
                        
                # Tempo
                if event.startswith('Tempo Value_'):
                    tempo_index = int(event.split('_')[1])
                    current_tempo = 30 + (tempo_index * 2)
                    midi.tempo_changes.append(
                        miditoolkit.midi.containers.TempoChange(
                            tempo=current_tempo,
                            time=int(current_time)  # Convert to int
                        )
                    )
                    
            except Exception as e:
                self.logger.error(f"Error processing event {event}: {str(e)}")
                continue
        
        # Add final note if exists
        if current_note is not None:
            notes_to_add.append(current_note)
        
        # Add notes to instrument
        instrument.notes = notes_to_add
        
        # Add instrument to MIDI file
        midi.instruments = [instrument]
        
        # Save MIDI file
        self.logger.info(f"Saving MIDI file with {len(notes_to_add)} notes")
        try:
            midi.dump(output_path)
            self.logger.info(f"Successfully saved MIDI file to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving MIDI file: {str(e)}")
            raise
        
    def midi_to_wav(self, midi_path: str, wav_path: str):
        """Convert MIDI to WAV using fluidsynth"""
        # You'll need to have fluidsynth installed and provide a soundfont
        soundfont_path = "path/to/your/soundfont.sf2"  # Update this path
        cmd = [
            'fluidsynth', 
            '-ni', 
            soundfont_path,
            midi_path, 
            '-F', 
            wav_path,
            '-r', '44100'
        ]
        subprocess.run(cmd)
        
    def generate_task1(self, 
                  num_samples: int = 20,
                  temperature: float = 0.95,
                  top_k: int = 32,
                  debug: bool = True):
        """Generate samples for Task 1"""
        self.logger.info("Generating samples for Task 1...")
        
        # Try one sample first in debug mode
        if debug:
            self.logger.info("Running in debug mode - generating one sample")
            num_samples = 1
        
        # Directory paths
        midi_dir = os.path.join(self.output_dir, 'task1', 'midi', f'temp_{temperature}')
        wav_dir = os.path.join(self.output_dir, 'task1', 'wav', f'temp_{temperature}')
        os.makedirs(midi_dir, exist_ok=True)
        os.makedirs(wav_dir, exist_ok=True)
        
        for i in tqdm(range(num_samples)):
            try:
                # Create a more structured prompt with a full bar
                prompt = torch.tensor([[
                    # Initialize sequence
                    self.preprocessor.special_tokens['BOS'],
                    self.preprocessor.special_tokens['BAR'],
                    
                    # Set initial tempo (keep it stable)
                    self.event2word['Position_1/16'],
                    self.event2word['Tempo Class_mid'],
                    self.event2word['Tempo Value_30'],
                    
                    # First bar - C major chord
                    self.event2word['Note On_60'],  # C4
                    self.event2word['Note Duration_4'],
                    self.event2word['Position_3/16'],
                    
                    self.event2word['Note On_64'],  # E4
                    self.event2word['Note Duration_4'],
                    self.event2word['Position_5/16'],
                    
                    self.event2word['Note On_67'],  # G4
                    self.event2word['Note Duration_4'],
                    self.event2word['Position_7/16'],
                    
                    # End first bar
                    self.preprocessor.special_tokens['BAR'],
                    
                    # Start second bar
                    self.event2word['Position_1/16'],
                    self.event2word['Note On_65'],  # F4
                    self.event2word['Note Duration_4'],
                    self.event2word['Position_3/16'],
                    
                    self.event2word['Note On_69'],  # A4
                    self.event2word['Note Duration_4'],
                    self.preprocessor.special_tokens['BAR'],
                ]], device=self.device)
                
                # Generate with basic parameters
                generated = self.model.generate(
                    prompt=prompt,
                    max_length=4096,  # Longer sequence
                    temperature=temperature,
                    top_k=top_k
                )
                
                # Convert to MIDI
                midi_path = os.path.join(midi_dir, f'sample_{i+1}.mid')
                self.tokens_to_midi(generated[0].cpu().tolist(), midi_path)
                
            except Exception as e:
                self.logger.error(f"Error during generation: {str(e)}")
                if debug:
                    raise e
                continue

    def generate_task2(self, 
                      prompt_paths: List[str],
                      temperatures: List[float] = [0.8, 1.0, 1.2],
                      top_k: int = 40):
        """Generate continuations for Task 2"""
        self.logger.info("Generating continuations for Task 2...")
        
        for i, prompt_path in enumerate(prompt_paths):
            self.logger.info(f"Processing prompt {i+1}")
            prompt_name = f'prompt{i+1}'
            
            # Load and process prompt
            prompt_events = self.preprocessor.midi_to_events(prompt_path)
            prompt_tokens = self.preprocessor.events_to_tokens(prompt_events)
            prompt_tensor = torch.tensor([prompt_tokens], device=self.device)
            
            for temp in temperatures:
                self.logger.info(f"Generating with temperature {temp}")
                
                # Directory paths
                midi_dir = os.path.join(self.output_dir, 'task2', 'midi', prompt_name)
                wav_dir = os.path.join(self.output_dir, 'task2', 'wav', prompt_name)
                
                # Generate continuation
                with torch.no_grad(), autocast():
                    generated = self.model.generate(
                        prompt=prompt_tensor,
                        max_length=self.config['generation']['max_length'],
                        temperature=temp,
                        top_k=top_k
                    )
                
                # Convert to MIDI
                midi_path = os.path.join(midi_dir, f'continuation_temp_{temp}.mid')
                self.tokens_to_midi(generated[0].cpu().tolist(), midi_path)
                
                # Convert to WAV
                wav_path = os.path.join(wav_dir, f'continuation_temp_{temp}.wav')
                self.midi_to_wav(midi_path, wav_path)

def main():
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='outputs',
                      help='Output directory')
    parser.add_argument('--task', type=int, choices=[1, 2], required=True,
                      help='Task to generate (1 or 2)')
    parser.add_argument('--prompt_dir', type=str,
                      help='Directory containing prompt files for Task 2')
    args = parser.parse_args()
    
    # Initialize generator
    generator = MusicGenerator(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_dir=args.output_dir
    )
    
    # Generate based on task
    if args.task == 1:
        generator.generate_task1()
    else:
        # Get prompt paths
        prompt_paths = sorted([
            os.path.join(args.prompt_dir, f) 
            for f in os.listdir(args.prompt_dir) 
            if f.endswith('.mid')
        ])
        generator.generate_task2(prompt_paths)

if __name__ == "__main__":
    main()