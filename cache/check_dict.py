import pickle

# Load dictionary
with open('/home/paperspace/dlmusic/hw3/tutorial/basic_event_dictionary.pkl', 'rb') as f:
    event2word, word2event = pickle.load(f)

# Group events by type
event_types = {}
for event in event2word.keys():
    event_type = event.split('_')[0] if '_' in event else event.split()[0]
    if event_type not in event_types:
        event_types[event_type] = []
    event_types[event_type].append(event)

# Print summary of each event type
print("\nEvent types in dictionary:")
for event_type, events in event_types.items():
    print(f"\n{event_type}:")
    print(f"Count: {len(events)}")
    print(f"Examples: {events[:3]}")