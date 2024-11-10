from miditoolkit import MidiFile
import sys

def check_midi(midi_path):
    print(f"\nChecking MIDI file: {midi_path}")
    midi = MidiFile(midi_path)
    
    print("MIDI file info:")
    print(f"Number of instruments: {len(midi.instruments)}")
    if len(midi.instruments) > 0:
        print(f"Number of notes in first instrument: {len(midi.instruments[0].notes)}")
    print(f"Tempo changes: {len(midi.tempo_changes)}")
    
    # Safely check time signatures
    time_sigs = getattr(midi, 'time_signatures', [])
    print(f"Time signatures: {len(time_sigs)}")
    
    # Print more details about instruments
    for i, inst in enumerate(midi.instruments):
        print(f"\nInstrument {i}:")
        print(f"  Program: {inst.program}")
        print(f"  Is drum: {inst.is_drum}")
        print(f"  Name: {inst.name}")
        print(f"  Notes: {len(inst.notes)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_midi.py <midi_file>")
        sys.exit(1)
    
    check_midi(sys.argv[1])
