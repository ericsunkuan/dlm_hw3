import pickle
with open('/home/paperspace/dlmusic/hw3/tutorial/basic_event_dictionary.pkl', 'rb') as f:
    event2word, word2event = pickle.load(f)
print("Number of events:", len(event2word))
print("First 10 events:", list(event2word.keys())[:10])

from miditoolkit import MidiFile

midi_path = "/home/paperspace/dlmusic/hw3/src/data/Pop1K7/midi_synchronized/src_003/746.mid"
midi = MidiFile(midi_path)
print("Number of tracks:", len(midi.instruments))
if len(midi.instruments) > 0:
    print("Notes in first track:", len(midi.instruments[0].notes))
print("Tempo changes:", len(midi.tempo_changes))