import pickle
import pretty_midi
import glob
import os
from collections import defaultdict

def analyze_midi_files(data_dir):
    """Analyze MIDI files in the dataset"""
    print("\n=== MIDI File Analysis ===")
    
    files = glob.glob(os.path.join(data_dir, "**/*.mid"), recursive=True)
    print(f"Total MIDI files found: {len(files)}")
    
    stats = {
        'lengths': [],
        'note_counts': [],
        'tempos': [],
        'time_signatures': [],
        'pitch_ranges': []
    }
    
    for f in files[:10]:  # Analyze first 10 files for quick stats
        try:
            pm = pretty_midi.PrettyMIDI(f)
            stats['lengths'].append(pm.get_end_time())
            note_count = sum(len(i.notes) for i in pm.instruments)
            stats['note_counts'].append(note_count)
            stats['tempos'].extend([t.tempo for t in pm.tempo_changes])
            stats['time_signatures'].extend([f"{ts.numerator}/{ts.denominator}" for ts in pm.time_signature_changes])
            
            all_notes = [note.pitch for inst in pm.instruments for note in inst.notes]
            if all_notes:
                stats['pitch_ranges'].append((min(all_notes), max(all_notes)))
            
            print(f"\nFile: {os.path.basename(f)}")
            print(f"Length: {pm.get_end_time():.2f} seconds")
            print(f"Notes: {note_count}")
            print(f"Tempo changes: {len(pm.tempo_changes)}")
            print(f"Time signatures: {[f'{ts.numerator}/{ts.denominator}' for ts in pm.time_signature_changes]}")
            
        except Exception as e:
            print(f"Error processing {f}: {str(e)}")
    
    return stats

def analyze_dictionary(dict_path):
    """Analyze the event dictionary"""
    print("\n=== Dictionary Analysis ===")
    
    try:
        with open(dict_path, 'rb') as f:
            dictionary = pickle.load(f)
        
        # Handle different dictionary formats
        if isinstance(dictionary, tuple):
            event2word = dictionary[0]
        elif isinstance(dictionary, dict):
            event2word = dictionary
        else:
            print(f"Unknown dictionary format: {type(dictionary)}")
            return
        
        print(f"Dictionary size: {len(event2word)}")
        
        # Analyze event types
        event_types = defaultdict(list)
        for event in event2word:
            try:
                type_ = event.split('_')[0]
                value = '_'.join(event.split('_')[1:])
                event_types[type_].append(value)
            except:
                print(f"Couldn't parse event: {event}")
        
        print("\nEvent types:")
        for type_, values in event_types.items():
            print(f"{type_}: {len(values)} unique values")
            print(f"Example values: {values[:5]}")
            
    except Exception as e:
        print(f"Error analyzing dictionary: {str(e)}")

if __name__ == "__main__":
    # Analyze MIDI files
    midi_dir = "/home/paperspace/dlmusic/hw3/src/data/Pop1K7/midi_analyzed"
    stats = analyze_midi_files(midi_dir)
    
    # Analyze dictionary
    dict_path = "cache/dictionary.pkl"
    analyze_dictionary(dict_path)