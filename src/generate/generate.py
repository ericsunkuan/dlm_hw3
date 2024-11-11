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
import pretty_midi

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
        
    def tokens_to_midi(self, tokens, output_path):
        """Convert token sequence to MIDI with simplified tempo handling"""
        self.logger.info(f"Converting tokens to MIDI: {output_path}")
        
        # Create PrettyMIDI object with fixed tempo
        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)  # Piano
        
        current_time = 0.0
        current_tempo = 120.0  # Default tempo
        current_position = 0
        notes_to_add = []
        
        # Add a time signature
        pm.time_signature_changes.append(
            pretty_midi.TimeSignature(numerator=4, denominator=4, time=0)
        )
        
        for token in tokens:
            event = self.word2event[token]
            if '_' not in event:
                continue
                
            event_type, value = event.split('_', 1)
            
            if event_type == 'Position':
                if '/' in value:
                    num, den = map(int, value.split('/'))
                    current_position = num / int(den)
                    # Convert position to time in seconds
                    bar_duration = 4 * 60.0 / current_tempo  # Length of one bar in seconds
                    current_time = current_position * bar_duration
                    
            elif event_type == 'Note On':
                try:
                    pitch = int(value)
                    if 21 <= pitch <= 108:  # Valid piano range
                        notes_to_add.append({
                            'pitch': pitch,
                            'start': current_time,
                            'velocity': 64  # Default velocity
                        })
                except ValueError:
                    continue
                
            elif event_type == 'Note Duration':
                if notes_to_add:
                    try:
                        duration = float(value) * 60.0 / (current_tempo * 4)  # Convert to seconds
                        for note in notes_to_add:
                            note_end = note['start'] + max(0.1, duration)  # Minimum duration
                            new_note = pretty_midi.Note(
                                velocity=note['velocity'],
                                pitch=note['pitch'],
                                start=note['start'],
                                end=note_end
                            )
                            instrument.notes.append(new_note)
                    except ValueError:
                        continue
                    notes_to_add = []
                    
            elif event_type == 'Note Velocity':
                try:
                    velocity = min(127, max(1, int(value) * 8))  # Ensure valid MIDI velocity
                    for note in notes_to_add:
                        note['velocity'] = velocity
                except ValueError:
                    continue
        
        # Add the instrument
        pm.instruments.append(instrument)
        
        # Sort notes by start time
        for inst in pm.instruments:
            inst.notes.sort(key=lambda x: x.start)
        
        # Write the MIDI file
        try:
            pm.write(output_path)
            self.logger.info(f"Saving MIDI file with {len(instrument.notes)} notes")
        except Exception as e:
            self.logger.error(f"Error writing MIDI file: {str(e)}")
            # Try to save with minimal settings
            pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
            pm.instruments.append(instrument)
            pm.write(output_path)
        
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
                  temperature: float = 0.8,    # Lower temperature
                  top_k: int = 16,             # More focused sampling
                  debug: bool = True):
        """Generate samples for Task 1"""
        self.logger.info("Generating samples for Task 1...")
        
        # Create a simpler prompt
        base_pattern = [
            # Bar start
            self.event2word['Position_1/16'],
            self.event2word['Note On_60'],  # C4
            self.event2word['Note Velocity_12'],
            self.event2word['Note Duration_2'],
            
            self.event2word['Position_5/16'],
            self.event2word['Note On_64'],  # E4
            self.event2word['Note Velocity_12'],
            self.event2word['Note Duration_2'],
            
            self.event2word['Position_9/16'],
            self.event2word['Note On_67'],  # G4
            self.event2word['Note Velocity_12'],
            self.event2word['Note Duration_2'],
            
            self.event2word['Bar_None']
        ]
        
        # Use a shorter prompt
        prompt = torch.tensor([base_pattern], device=self.device)
        
        try:
            # Generate with simpler parameters
            generated = self.model.generate(
                prompt=prompt,
                max_length=1024,  # Shorter sequence
                temperature=temperature,
                top_k=top_k
            )
            
            # Convert to MIDI
            midi_path = os.path.join(
                self.output_dir, 'task1', 'midi', 
                f'temp_{temperature}', 'sample_1.mid'
            )
            os.makedirs(os.path.dirname(midi_path), exist_ok=True)
            self.tokens_to_midi(generated[0].cpu().tolist(), midi_path)
            
        except Exception as e:
            self.logger.error(f"Error during generation: {str(e)}")
            if debug:
                raise e
        
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