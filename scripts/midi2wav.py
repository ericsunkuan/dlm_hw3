import os
import pretty_midi
import numpy as np
from scipy.io import wavfile
import argparse
from tqdm import tqdm

def convert_midi_to_wav(midi_path: str, output_path: str, sample_rate: int = 44100):
    """Convert MIDI to WAV using pretty_midi"""
    try:
        print(f"\nConverting: {os.path.basename(midi_path)}")
        
        # Load MIDI file
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        print(f"MIDI length: {midi_data.get_end_time():.2f} seconds")
        
        # Synthesize audio
        audio_data = midi_data.synthesize(fs=sample_rate)
        
        # Normalize audio
        audio_data = np.int16(audio_data * 32767)
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save WAV file
        wavfile.write(output_path, sample_rate, audio_data)
        print(f"Saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error converting {midi_path}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert MIDI files to WAV')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Directory containing MIDI files')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Output directory for WAV files')
    args = parser.parse_args()
    
    # Find all MIDI files
    midi_files = []
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith('.mid'):
                midi_files.append(os.path.join(root, file))
    
    print(f"Found {len(midi_files)} MIDI files")
    
    # Convert each file
    success_count = 0
    for midi_path in tqdm(midi_files, desc="Converting files"):
        # Create parallel directory structure
        rel_path = os.path.relpath(midi_path, args.input_dir)
        wav_path = os.path.join(args.output_dir, rel_path.replace('.mid', '.wav'))
        
        # Convert
        if convert_midi_to_wav(midi_path, wav_path):
            success_count += 1
    
    print(f"\nSuccessfully converted {success_count}/{len(midi_files)} files")

if __name__ == "__main__":
    main()