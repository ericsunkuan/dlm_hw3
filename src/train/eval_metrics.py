import numpy as np
from glob import glob
import random, itertools
import pickle
import pandas as pd
import os
import scipy.stats
import tqdm
from tqdm import tqdm
import argparse
from miditoolkit import MidiFile

import sys
# Add the parent directory to the path so we can import utils
sys.path.append('/home/paperspace/dlmusic/hw3/src/train')
import utils  # Import local utils.py

sys.path.append('/home/paperspace/dlmusic/hw3/MusDr')
from musdr.side_utils import (
    get_bars_crop, 
    get_pitch_histogram, 
    compute_histogram_entropy, 
    get_onset_xor_distance,
    get_chord_sequence,
    read_fitness_mat
)


#############################################################################
'''
Default event encodings (ones used by the Jazz Transformer).
You may override the defaults in function arguments to suit your own vocabulary.
'''
BAR_EV = 0                  # the ID of ``Bar`` event
POS_EVS = range(1, 17)      # the IDs of ``Position`` events
PITCH_EVS = range(99, 185)  # the ID of Pitch => Note on events
#############################################################################

# Add these as global variables at the top of the file
event2word = None
word2event = None

def parse_opt():
    parser = argparse.ArgumentParser()
    # training opts
    parser.add_argument('--dict_path', type=str,
                        help='the dictionary path', required=True)
    parser.add_argument('--output_file_path', type=str,
                        help='the output file path.', required=True)
    args = parser.parse_args()
    return args

# Only execute this if the script is run directly
if __name__ == "__main__":
    opt = parse_opt()
else:
    # When imported as a module, initialize opt with None or default values
    class DummyOpt:
        def __init__(self):
            self.dict_path = None
            self.output_file_path = None
    opt = DummyOpt()

if __name__ == "__main__":
    event2word, word2event = pickle.load(open(opt.dict_path, 'rb'))


def extract_events(input_path: str, event2word: dict):
    """Extract events from MIDI file"""
    try:
        # Read MIDI file
        note_items, tempo_items = utils.read_items(input_path)
        print(f"\nProcessing {input_path}")
        print(f"Found {len(note_items)} notes and {len(tempo_items)} tempo changes")
        
        # Quantize items
        note_items = utils.quantize_items(note_items)
        tempo_items = utils.quantize_items(tempo_items)
        
        # Group items
        groups = utils.group_items(note_items, tempo_items)
        
        # Convert to events
        events = utils.item2event(groups, event2word)  # Pass event2word
        
        if events:
            print(f"Converted to {len(events)} events")
            print("First few events:", events[:5])
            return events
        return None
        
    except Exception as e:
        print(f"Error in extract_events for {input_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def prepare_data(midi_path: str, event2word: dict):
    """Prepare data from MIDI file"""
    events = extract_events(midi_path, event2word)
    if events is None:
        print(f"Warning: No valid tokens found for {midi_path}")
        return None
    return events

def compute_piece_pitch_entropy(piece_ev_seq, window_size=4):
    """Compute pitch-class histogram entropy"""
    try:
        # Extract pitches
        pitches = []
        for event in piece_ev_seq:
            if event.startswith('Note On_'):
                pitch = int(event.split('_')[1])
                pitches.append(pitch % 12)  # Convert to pitch class
        
        if not pitches:
            print("No pitches found in sequence")
            return None
        
        # Calculate entropy
        from collections import Counter
        import numpy as np
        pitch_counts = Counter(pitches)
        total = sum(pitch_counts.values())
        probabilities = [count/total for count in pitch_counts.values()]
        entropy = -sum(p * np.log2(p) for p in probabilities)
        
        print(f"Found {len(pitches)} notes, {len(set(pitches))} unique pitch classes")
        print(f"Entropy: {entropy:.4f}")
        
        return entropy
        
    except Exception as e:
        print(f"Error computing pitch entropy: {str(e)}")
        return None

def compute_piece_groove_similarity(piece_ev_seq):
    """Compute groove similarity based on position events"""
    try:
        # Extract all position events
        positions = []
        current_bar = []
        
        for event in piece_ev_seq:
            if event.startswith('Position_'):
                pos = event.split('_')[1].replace('/16', '')
                current_bar.append(int(pos))
                # When we see position 1, it's a new bar
                if pos == '1/16' and current_bar:
                    positions.append(current_bar)
                    current_bar = []
        
        # Add the last bar if not empty
        if current_bar:
            positions.append(current_bar)
        
        if len(positions) < 2:
            print("Not enough bars for groove similarity calculation")
            return 0.0  # Return 0 for no similarity
        
        # Calculate similarity between consecutive bars
        similarities = []
        for i in range(len(positions)-1):
            if positions[i] and positions[i+1]:  # Only compare non-empty bars
                common = len(set(positions[i]) & set(positions[i+1]))
                total = len(set(positions[i]) | set(positions[i+1]))
                if total > 0:
                    similarities.append(common / total)
        
        if not similarities:
            print("No valid pattern pairs for similarity calculation")
            return 0.0  # Return 0 for no similarity
        
        similarity = sum(similarities) / len(similarities)
        print(f"Found {len(positions)} bars")
        print(f"Average similarity: {similarity:.4f}")
        
        return similarity
        
    except Exception as e:
        print(f"Error computing groove similarity: {str(e)}")
        return 0.0  # Return 0 for error cases

def load_dictionary(dict_path: str):
    """Load event dictionary"""
    global event2word, word2event
    import pickle
    with open(dict_path, 'rb') as f:
        event2word, word2event = pickle.load(f)
    print(f"Loaded dictionary with {len(event2word)} events")
    return event2word, word2event

if __name__ == "__main__":
  # codes below are for testing
  test_pieces = sorted(glob(os.path.join(opt.output_file_path, '*.mid')))

  # print (test_pieces)

  result_dict = {
      'piece_name': [],
      'H1': [],
      'H4': [],
      'GS': []
  }

  for p in tqdm(test_pieces):
      result_dict['piece_name'].append(p.replace('\\', '/').split('/')[-1])
      seq = prepare_data(p, event2word)

      h1 = compute_piece_pitch_entropy(seq, 1)
      result_dict['H1'].append(h1)
      h4 = compute_piece_pitch_entropy(seq, 4)
      result_dict['H4'].append(h4)
      gs = compute_piece_groove_similarity(seq)
      result_dict['GS'].append(gs)

  if len(result_dict):
      df = pd.DataFrame.from_dict(result_dict)
      df.to_csv('pop1k7.csv', index=False, encoding='utf-8')