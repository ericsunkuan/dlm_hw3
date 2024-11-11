import pickle
from collections import defaultdict

def analyze_dictionary(dict_path):
    """Analyze the basic event dictionary"""
    print(f"\n=== Analyzing Dictionary: {dict_path} ===")
    
    try:
        with open(dict_path, 'rb') as f:
            dictionary = pickle.load(f)
            
        if isinstance(dictionary, tuple):
            event2word, word2event = dictionary
        else:
            print(f"Unknown dictionary format: {type(dictionary)}")
            return
            
        print(f"Dictionary size: {len(event2word)}")
        
        # Analyze event types
        event_types = defaultdict(list)
        for event in event2word:
            try:
                if '_' in event:
                    type_ = event.split('_')[0]
                    value = '_'.join(event.split('_')[1:])
                    event_types[type_].append(value)
                else:
                    event_types['special'].append(event)
            except:
                print(f"Couldn't parse event: {event}")
        
        print("\nEvent types:")
        for type_, values in event_types.items():
            print(f"\n{type_}: {len(values)} unique values")
            print(f"Example values: {sorted(values)[:5]}")
            
        # Print some specific event ranges
        if 'Note On' in event_types:
            note_values = sorted([int(v) for v in event_types['Note On']])
            print(f"\nNote range: {min(note_values)} to {max(note_values)}")
            
        if 'Note Duration' in event_types:
            dur_values = sorted([int(v) for v in event_types['Note Duration']])
            print(f"Duration range: {min(dur_values)} to {max(dur_values)}")
            
        if 'Position' in event_types:
            print(f"Position values: {sorted(event_types['Position'])[:5]}...")
            
        return event2word, word2event
            
    except Exception as e:
        print(f"Error analyzing dictionary: {str(e)}")
        return None, None

if __name__ == "__main__":
    dict_path = "/home/paperspace/dlmusic/hw3/cache/basic_event_dictionary.pkl"
    event2word, word2event = analyze_dictionary(dict_path)
    
    if event2word:
        print("\nDictionary looks valid!")
        
        # Test a few token conversions
        test_events = ['Note On_60', 'Position_1/16', 'BAR']
        print("\nTesting token conversion:")
        for event in test_events:
            if event in event2word:
                word = event2word[event]
                back_to_event = word2event[word]
                print(f"{event} -> {word} -> {back_to_event}")
            else:
                print(f"Event not found: {event}")