import os
import pandas as pd
from tqdm import tqdm
import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.train.eval_metrics import prepare_data, compute_piece_pitch_entropy, compute_piece_groove_similarity, load_dictionary

def evaluate_generated_samples(generated_dir: str, dict_path: str):
    """Evaluate only the generated MIDI files"""
    results = {
        'file': [],
        'H4': [],
        'GS': []
    }
    
    # Load dictionary
    event2word, _ = load_dictionary(dict_path)
    
    # Find all generated MIDI files
    midi_files = []
    for root, _, files in os.walk(generated_dir):
        for file in files:
            if file.endswith('.mid'):
                midi_files.append(os.path.join(root, file))
    
    print(f"Found {len(midi_files)} generated MIDI files")
    
    # Process each file
    for midi_path in tqdm(midi_files, desc="Processing generated files"):
        try:
            # Prepare data
            seq = prepare_data(midi_path, event2word)
            if seq is None:
                print(f"Warning: Could not process {midi_path}")
                continue
            
            print(f"\nProcessing: {os.path.basename(midi_path)}")
            print(f"Sequence length: {len(seq)}")
            print("First few events:", seq[:10])
            
            # Calculate metrics
            h4 = compute_piece_pitch_entropy(seq, window_size=4)
            gs = compute_piece_groove_similarity(seq)
            
            # Store results (use 0.0 for None values)
            results['file'].append(os.path.basename(midi_path))
            results['H4'].append(h4 if h4 is not None else 0.0)
            results['GS'].append(gs if gs is not None else 0.0)
                
        except Exception as e:
            print(f"Error processing {midi_path}: {str(e)}")
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Print statistics
    print("\nGenerated Samples Statistics:")
    if len(df) > 0:
        print("\nH4 (Pitch Entropy):")
        print(f"Mean: {df['H4'].mean():.4f}")
        print(f"Std: {df['H4'].std():.4f}")
        print(f"Min: {df['H4'].min():.4f}")
        print(f"Max: {df['H4'].max():.4f}")
        
        print("\nGS (Groove Similarity):")
        print(f"Mean: {df['GS'].mean():.4f}")
        print(f"Std: {df['GS'].std():.4f}")
        print(f"Min: {df['GS'].min():.4f}")
        print(f"Max: {df['GS'].max():.4f}")
    else:
        print("No valid samples were processed")
    
    # Save results
    output_path = os.path.join(os.path.dirname(generated_dir), 'generated_evaluation.csv')
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated_dir', type=str, required=True,
                      help='Directory containing generated MIDI files')
    parser.add_argument('--dict_path', type=str, required=True,
                      help='Path to event dictionary')
    args = parser.parse_args()
    
    evaluate_generated_samples(args.generated_dir, args.dict_path)

if __name__ == "__main__":
    main()