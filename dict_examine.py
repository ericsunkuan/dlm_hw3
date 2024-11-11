import pickle
with open('cache/dictionary.pkl', 'rb') as f:
    event2word, word2event = pickle.load(f)
print("Dictionary size:", len(event2word))
print("\nEvent types:")
event_types = {}
for event in event2word:
    type_ = event.split('_')[0]
    event_types[type_] = event_types.get(type_, 0) + 1
for type_, count in event_types.items():
    print(f"{type_}: {count} events")