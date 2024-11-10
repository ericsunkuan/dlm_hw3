import chord_recognition
import numpy as np
import miditoolkit

# parameters for input
DEFAULT_VELOCITY_BINS = np.linspace(0, 128, 32+1, dtype=np.int32)
DEFAULT_FRACTION = 16
DEFAULT_DURATION_BINS = np.arange(60, 3841, 60, dtype=int)
DEFAULT_TEMPO_INTERVALS = [range(30, 90), range(90, 150), range(150, 210)]

# parameters for output
DEFAULT_RESOLUTION = 480

# define "Item" for general storage
class Item(object):
    def __init__(self, name, start, end, velocity, pitch):
        self.name = name
        self.start = start
        self.end = end
        self.velocity = velocity
        self.pitch = pitch

    def __repr__(self):
        return 'Item(name={}, start={}, end={}, velocity={}, pitch={})'.format(
            self.name, self.start, self.end, self.velocity, self.pitch)

# read notes and tempo changes from midi (assume there is only one track)
def read_items(file_path):
    """Read items from MIDI file with better tempo handling"""
    midi_obj = miditoolkit.midi.parser.MidiFile(file_path)
    
    # Note items
    note_items = []
    for note in midi_obj.instruments[0].notes:
        note_items.append(
            Item(
                name='Note',
                start=note.start,
                end=note.end,
                velocity=note.velocity,
                pitch=note.pitch
            )
        )
    
    # Tempo items
    tempo_items = []
    for tempo in midi_obj.tempo_changes:
        tempo_items.append(
            Item(
                name='Tempo',
                start=tempo.time,
                end=None,  # Tempo changes don't have end times
                velocity=0,  # Not applicable for tempo
                pitch=int(tempo.tempo)  # Store tempo as pitch
            )
        )
    
    return note_items, tempo_items

# quantize items
def quantize_items(items):
    """Quantize items with better handling of tempo events"""
    if not items:
        return []
        
    # Find the first note or tempo change
    start_time = min(item.start for item in items)
    
    # Quantize all items
    ticks = 32  # Can be adjusted
    for item in items:
        shift = (item.start - start_time) % ticks
        item.start += (ticks - shift) if shift else 0
        
        # Only adjust end time for notes (tempo changes might not have end time)
        if hasattr(item, 'end') and item.end is not None:
            shift = (item.end - start_time) % ticks
            item.end += (ticks - shift) if shift else 0
    
    return items

# extract chord
def extract_chords(items):
    method = chord_recognition.MIDIChord()
    chords = method.extract(notes=items)
    output = []
    for chord in chords:
        output.append(Item(
            name='Chord',
            start=chord[0],
            end=chord[1],
            velocity=None,
            pitch=chord[2].split('/')[0]))
    return output

# group items
def group_items(note_items, tempo_items):
    """Group items with better tempo handling"""
    # Sort items by start time
    note_items.sort(key=lambda x: x.start)
    tempo_items.sort(key=lambda x: x.start)
    
    # Create groups
    groups = []
    current_bar = 0
    current_position = 0
    
    # Add items to groups
    for note in note_items:
        # Add bar marker if needed
        if note.start >= (current_bar + 1) * DEFAULT_RESOLUTION * 4:
            current_bar = note.start // (DEFAULT_RESOLUTION * 4)
            groups.append({'bar': True})
        
        # Add position
        position = (note.start % (DEFAULT_RESOLUTION * 4)) / (DEFAULT_RESOLUTION * 4)
        if position != current_position:
            current_position = position
            groups.append({'position': position})
        
        # Add note
        if not groups or 'notes' not in groups[-1]:
            groups.append({'notes': []})
        groups[-1]['notes'].append(note)
    
    # Add tempo changes
    for tempo in tempo_items:
        # Find the right position to insert tempo
        for i, group in enumerate(groups):
            if 'notes' in group and group['notes'][0].start > tempo.start:
                groups.insert(i, {'tempo': tempo.pitch})
                break
    
    return groups

# define "Event" for event storage
class Event(object):
    def __init__(self, name, time, value, text):
        self.name = name
        self.time = time
        self.value = value
        self.text = text

    def __repr__(self):
        return 'Event(name={}, time={}, value={}, text={})'.format(
            self.name, self.time, self.value, self.text)

# item to event
def item2event(groups, event2word):
    """Convert items to events using the correct dictionary format"""
    events = []
    n_downbeat = 0
    
    for i in range(len(groups)):
        if 'bar' in groups[i]:
            events.append('Bar_None')
            n_downbeat += 1
        
        if 'position' in groups[i]:
            position = float(groups[i]['position'])
            position_index = int(position * 16) + 1
            position_event = f'Position_{position_index}/16'
            if position_event in event2word:
                events.append(position_event)
        
        if 'notes' in groups[i]:
            for note in groups[i]['notes']:
                # Note velocity (quantize to available values)
                velocity_event = f'Note Velocity_{note.velocity}'
                if velocity_event in event2word:
                    events.append(velocity_event)
                
                # Note pitch becomes Note On
                note_event = f'Note On_{note.pitch}'
                if note_event in event2word:
                    events.append(note_event)
                
                # Calculate note duration
                duration = note.end - note.start
                duration_index = min(63, int(duration / 120))  # Quantize duration
                duration_event = f'Note Duration_{duration_index}'
                if duration_event in event2word:
                    events.append(duration_event)
        
        if 'tempo' in groups[i]:
            tempo = groups[i]['tempo']
            # Convert tempo to class
            if tempo < 90:
                tempo_class = 'slow'
            elif tempo < 120:
                tempo_class = 'mid'
            else:
                tempo_class = 'fast'
            tempo_class_event = f'Tempo Class_{tempo_class}'
            
            # Convert tempo to value index (0-59)
            tempo_value = min(59, max(0, int((tempo - 30) / 2)))
            tempo_value_event = f'Tempo Value_{tempo_value}'
            
            if tempo_class_event in event2word:
                events.append(tempo_class_event)
            if tempo_value_event in event2word:
                events.append(tempo_value_event)
    
    return events

#############################################################################################
# WRITE MIDI
#############################################################################################
def word_to_event(words, word2event):
    events = []
    for word in words:
        event_name, event_value = word2event.get(word).split('_')
        events.append(Event(event_name, None, event_value, None))
    return events

def write_midi(words, word2event, output_path, prompt_path=None):
    events = word_to_event(words, word2event)
    # get downbeat and note (no time)
    temp_notes = []
    temp_chords = []
    temp_tempos = []
    for i in range(len(events)-3):
        if events[i].name == 'Bar' and i > 0:
            temp_notes.append('Bar')
            temp_chords.append('Bar')
            temp_tempos.append('Bar')
        elif events[i].name == 'Position' and \
            events[i+1].name == 'Note Velocity' and \
            events[i+2].name == 'Note On' and \
            events[i+3].name == 'Note Duration':
            # start time and end time from position
            position = int(events[i].value.split('/')[0]) - 1
            # velocity
            index = int(events[i+1].value)
            velocity = int(DEFAULT_VELOCITY_BINS[index])
            # pitch
            pitch = int(events[i+2].value)
            # duration
            index = int(events[i+3].value)
            duration = DEFAULT_DURATION_BINS[index]
            # adding
            temp_notes.append([position, velocity, pitch, duration])
        elif events[i].name == 'Position' and events[i+1].name == 'Chord':
            position = int(events[i].value.split('/')[0]) - 1
            temp_chords.append([position, events[i+1].value])
        elif events[i].name == 'Position' and \
            events[i+1].name == 'Tempo Class' and \
            events[i+2].name == 'Tempo Value':
            position = int(events[i].value.split('/')[0]) - 1
            if events[i+1].value == 'slow':
                tempo = DEFAULT_TEMPO_INTERVALS[0].start + int(events[i+2].value)
            elif events[i+1].value == 'mid':
                tempo = DEFAULT_TEMPO_INTERVALS[1].start + int(events[i+2].value)
            elif events[i+1].value == 'fast':
                tempo = DEFAULT_TEMPO_INTERVALS[2].start + int(events[i+2].value)
            temp_tempos.append([position, tempo])
    # get specific time for notes
    ticks_per_beat = DEFAULT_RESOLUTION
    ticks_per_bar = DEFAULT_RESOLUTION * 4 # assume 4/4
    notes = []
    current_bar = 0
    for note in temp_notes:
        if note == 'Bar':
            current_bar += 1
        else:
            position, velocity, pitch, duration = note
            # position (start time)
            current_bar_st = current_bar * ticks_per_bar
            current_bar_et = (current_bar + 1) * ticks_per_bar
            flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
            st = flags[position]
            # duration (end time)
            et = st + duration
            notes.append(miditoolkit.Note(velocity, pitch, st, et))
    # get specific time for chords
    if len(temp_chords) > 0:
        chords = []
        current_bar = 0
        for chord in temp_chords:
            if chord == 'Bar':
                current_bar += 1
            else:
                position, value = chord
                # position (start time)
                current_bar_st = current_bar * ticks_per_bar
                current_bar_et = (current_bar + 1) * ticks_per_bar
                flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
                st = flags[position]
                chords.append([st, value])
    # get specific time for tempos
    tempos = []
    current_bar = 0
    for tempo in temp_tempos:
        if tempo == 'Bar':
            current_bar += 1
        else:
            position, value = tempo
            # position (start time)
            current_bar_st = current_bar * ticks_per_bar
            current_bar_et = (current_bar + 1) * ticks_per_bar
            flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
            st = flags[position]
            tempos.append([int(st), value])
    # write
    if prompt_path:
        midi = miditoolkit.midi.parser.MidiFile(prompt_path)
        #
        last_time = DEFAULT_RESOLUTION * 4 * 4
        # note shift
        for note in notes:
            note.start += last_time
            note.end += last_time
        midi.instruments[0].notes.extend(notes)
        # tempo changes
        temp_tempos = []
        for tempo in midi.tempo_changes:
            if tempo.time < DEFAULT_RESOLUTION*4*4:
                temp_tempos.append(tempo)
            else:
                break
        for st, bpm in tempos:
            st += last_time
            temp_tempos.append(miditoolkit.midi.containers.TempoChange(bpm, st))
        midi.tempo_changes = temp_tempos
        # write chord into marker
        if len(temp_chords) > 0:
            for c in chords:
                midi.markers.append(
                    miditoolkit.midi.containers.Marker(text=c[1], time=c[0]+last_time))
    else:
        midi = miditoolkit.midi.parser.MidiFile()
        midi.ticks_per_beat = DEFAULT_RESOLUTION
        # write instrument
        inst = miditoolkit.midi.containers.Instrument(0, is_drum=False)
        inst.notes = notes
        midi.instruments.append(inst)
        # write tempo
        tempo_changes = []
        for st, bpm in tempos:
            tempo_changes.append(miditoolkit.midi.containers.TempoChange(bpm, st))
        midi.tempo_changes = tempo_changes
        # write chord into marker
        if len(temp_chords) > 0:
            for c in chords:
                midi.markers.append(
                    miditoolkit.midi.containers.Marker(text=c[1], time=c[0]))
    # write
    midi.dump(output_path)
