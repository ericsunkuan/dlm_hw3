import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src/train to path for utils.py
sys.path.append('/home/paperspace/dlmusic/hw3/src/train')

# Add MusDr to path
sys.path.append('/home/paperspace/dlmusic/hw3/MusDr')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.train.eval_metrics import (
    prepare_data,
    compute_piece_pitch_entropy,
    compute_piece_groove_similarity,
    load_dictionary
)

def evaluate_directory(dir_path, dict_path, batch_size=50):
    """Evaluate all MIDI files in a directory with better error handling"""
    results = {
        'file': [],
        'H4': [],
        'GS': []
    }
    
    # Load dictionary first
    load_dictionary(dict_path)
    
    # Find all MIDI files
    midi_files = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.mid'):
                midi_files.append(os.path.join(root, file))
    
    print(f"Found {len(midi_files)} MIDI files")
    if len(midi_files) == 0:
        print(f"No MIDI files found in {dir_path}")
        return pd.DataFrame(results)
    
    # Process files in batches
    for i in range(0, len(midi_files), batch_size):
        batch = midi_files[i:i+batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{len(midi_files)//batch_size + 1}")
        
        batch_results = {
            'file': [],
            'H4': [],
            'GS': []
        }
        
        # Process each file in the batch
        for midi_path in tqdm(batch, desc=f"Batch {i//batch_size + 1}"):
            try:
                # Prepare data
                seq = prepare_data(midi_path)
                if seq is None:
                    continue
                    
                # Calculate metrics
                h4 = compute_piece_pitch_entropy(seq, window_size=4)
                gs = compute_piece_groove_similarity(seq)
                
                # Only store results if both metrics are valid
                if h4 is not None and gs is not None:
                    batch_results['file'].append(os.path.basename(midi_path))
                    batch_results['H4'].append(h4)
                    batch_results['GS'].append(gs)
                
            except KeyboardInterrupt:
                print("\nInterrupted by user. Saving current results...")
                break
            except Exception as e:
                print(f"Error processing {midi_path}: {str(e)}")
                continue
        
        # Save batch results
        results['file'].extend(batch_results['file'])
        results['H4'].extend(batch_results['H4'])
        results['GS'].extend(batch_results['GS'])
        
        # Save intermediate results
        df = pd.DataFrame(results)
        df.to_csv(f'evaluation_results/intermediate_results_batch_{i//batch_size + 1}.csv', index=False)
        print(f"Saved intermediate results: {len(df)} files processed so far")
        
        if KeyboardInterrupt:
            break
    
    print(f"Successfully processed {len(results['file'])} files total")
    return pd.DataFrame(results)

def plot_comparison(gen_df, ref_df, output_dir):
    """Plot comparison of metrics between generated and reference samples"""
    if len(gen_df) == 0 and len(ref_df) == 0:
        print("No data to plot - both dataframes are empty")
        return
        
    metrics = ['H4', 'GS']
    plt.figure(figsize=(15, 5))
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 2, i)
        
        if len(ref_df) > 0:
            plt.hist(ref_df[metric].dropna(), alpha=0.5, label='Reference', bins=20)
        if len(gen_df) > 0:
            plt.hist(gen_df[metric].dropna(), alpha=0.5, label='Generated', bins=20)
        
        plt.title(f'{metric} Distribution')
        plt.xlabel(metric)
        plt.ylabel('Count')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate generated music using MusDr metrics')
    parser.add_argument('--generated_dir', type=str, required=True,
                      help='Directory containing generated MIDI files')
    parser.add_argument('--reference_dir', type=str, required=True,
                      help='Directory containing reference MIDI files')
    parser.add_argument('--dict_path', type=str, required=True,
                      help='Path to the event dictionary')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                      help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=50,
                      help='Number of files to process in each batch')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Evaluating generated samples...")
    gen_df = evaluate_directory(args.generated_dir, args.dict_path, args.batch_size)
    
    print("\nEvaluating reference samples...")
    ref_df = evaluate_directory(args.reference_dir, args.dict_path, args.batch_size)
    
    # Save final results
    gen_df.to_csv(os.path.join(args.output_dir, 'generated_metrics.csv'), index=False)
    ref_df.to_csv(os.path.join(args.output_dir, 'reference_metrics.csv'), index=False)
    
    # Plot comparisons
    plot_comparison(gen_df, ref_df, args.output_dir)
    
    # Calculate and print overall statistics
    print("\nOverall Statistics:")
    for metric in ['H4', 'GS']:
        print(f"\n{metric}:")
        if len(ref_df) > 0:
            print(f"Reference - Mean: {ref_df[metric].mean():.4f}, Std: {ref_df[metric].std():.4f}")
        if len(gen_df) > 0:
            print(f"Generated - Mean: {gen_df[metric].mean():.4f}, Std: {gen_df[metric].std():.4f}")
        
        # Calculate similarity between distributions only if both have data
        if len(ref_df) > 0 and len(gen_df) > 0:
            from scipy import stats
            ks_stat, p_value = stats.ks_2samp(ref_df[metric], gen_df[metric])
            print(f"KS test - statistic: {ks_stat:.4f}, p-value: {p_value:.4f}")

if __name__ == '__main__':
    main()
