import os
import glob
import pretty_midi
import numpy as np
import pandas as pd
from collections import defaultdict
import pickle
from tqdm import tqdm

def load_dictionary(dict_path):
    """Load and verify the event dictionary"""
    with open(dict_path, 'rb') as f:
        dictionary = pickle.load(f)
    if isinstance(dictionary, tuple):
        event2word, word2event = dictionary
    else:
        raise ValueError("Invalid dictionary format")
    return event2word, word2event

def calculate_pitch_entropy(notes):
    """Calculate pitch class entropy"""
    if not notes:
        return 0.0
    
    # Convert to pitch classes (0-11)
    pitch_classes = [note % 12 for note in notes]
    
    # Calculate distribution
    counts = np.bincount(pitch_classes, minlength=12)
    probs = counts / len(pitch_classes)
    
    # Remove zeros before calculating entropy
    probs = probs[probs > 0]
    
    # Calculate entropy
    return -np.sum(probs * np.log2(probs))

def analyze_rhythm(pm, resolution=0.25):
    """Analyze rhythmic patterns"""
    if not pm.instruments or not pm.instruments[0].notes:
        return 0.0
    
    # Get note onset times
    onsets = [note.start for inst in pm.instruments for note in inst.notes]
    onsets = np.array(sorted(onsets))
    
    # Calculate inter-onset intervals
    iois = np.diff(onsets)
    
    # Quantize to resolution
    iois = np.round(iois / resolution) * resolution
    
    # Calculate IOI entropy
    unique_iois, counts = np.unique(iois, return_counts=True)
    probs = counts / len(iois)
    
    return -np.sum(probs * np.log2(probs))

def evaluate_midi(midi_path):
    """Evaluate a single MIDI file"""
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        
        # Basic stats
        duration = pm.get_end_time()
        notes = [note.pitch for inst in pm.instruments for note in inst.notes]
        n_notes = len(notes)
        
        if n_notes == 0:
            return None
        
        # Calculate metrics
        pitch_entropy = calculate_pitch_entropy(notes)
        rhythm_entropy = analyze_rhythm(pm)
        tempo_changes = len(pm.tempo_changes)
        avg_tempo = np.mean([t.tempo for t in pm.tempo_changes]) if pm.tempo_changes else 0
        
        return {
            'duration': duration,
            'n_notes': n_notes,
            'unique_pitches': len(set(notes)),
            'pitch_entropy': pitch_entropy,
            'rhythm_entropy': rhythm_entropy,
            'tempo_changes': tempo_changes,
            'avg_tempo': avg_tempo
        }
        
    except Exception as e:
        print(f"Error processing {midi_path}: {str(e)}")
        return None

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated_dir', type=str, required=True)
    parser.add_argument('--dict_path', type=str, required=True)
    args = parser.parse_args()
    
    # Load dictionary
    print(f"Loading dictionary from {args.dict_path}")
    event2word, word2event = load_dictionary(args.dict_path)
    print(f"Loaded dictionary with {len(event2word)} events")
    
    # Find MIDI files
    midi_files = glob.glob(os.path.join(args.generated_dir, '**/*.mid'), recursive=True)
    print(f"\nFound {len(midi_files)} generated MIDI files")
    
    # Evaluate each file
    results = []
    for midi_file in tqdm(midi_files, desc="Evaluating files"):
        metrics = evaluate_midi(midi_file)
        if metrics:
            metrics['file'] = os.path.basename(midi_file)
            results.append(metrics)
    
    # Create summary
    if results:
        df = pd.DataFrame(results)
        print("\nGeneration Statistics:")
        print(f"Number of valid files: {len(results)}")
        print("\nAverages:")
        print(f"Duration: {df['duration'].mean():.2f}s")
        print(f"Notes per file: {df['n_notes'].mean():.1f}")
        print(f"Unique pitches: {df['unique_pitches'].mean():.1f}")
        print(f"Pitch entropy: {df['pitch_entropy'].mean():.4f}")
        print(f"Rhythm entropy: {df['rhythm_entropy'].mean():.4f}")
        print(f"Tempo changes: {df['tempo_changes'].mean():.1f}")
        
        # Save results
        output_file = os.path.join(os.path.dirname(args.generated_dir), 'evaluation_results.csv')
        df.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to: {output_file}")
    else:
        print("No valid results found!")

if __name__ == "__main__":
    main()
