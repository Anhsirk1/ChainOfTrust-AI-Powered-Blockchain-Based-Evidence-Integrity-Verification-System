#!/usr/bin/env python3
"""
Batch processing example for FractalVideoGuard.

Demonstrates:
- Processing multiple videos
- Parallel execution
- Results aggregation
- CSV export
"""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
import csv
from fractalvideoguard_v0_5_2 import extract_features, ConfigPresets

def process_single_video(video_path, config):
    """Process a single video and return results."""
    try:
        features, debug = extract_features(str(video_path), config=config)
        
        # Extract key features
        result = {
            'filename': video_path.name,
            'success': True,
            'h_dfa': features['hurst_dfa'],
            'h_r2': features['hurst_dfa_r2'],
            'd_box': features['fractal_dim_box_mean'],
            'dct_hf': features['dct_hf_mean'],
            'ringing': features['ringing_mean'],
            'edge_density': features['edge_density_mean'],
            'n_samples': features['n_samples_used'],
            # Simple deepfake score
            'score_fake': abs(0.70 - features['hurst_dfa']) * 2.0 + 
                         abs(1.35 - features['fractal_dim_box_mean']) * 1.5,
            'likely_fake': features['hurst_dfa'] < 0.60 or 
                          features['fractal_dim_box_mean'] < 1.20
        }
        
        print(f"✅ {video_path.name}: H={result['h_dfa']:.3f}, D={result['d_box']:.3f}")
        return result
        
    except Exception as e:
        print(f"❌ {video_path.name}: {str(e)}")
        return {
            'filename': video_path.name,
            'success': False,
            'error': str(e)
        }

def batch_process_sequential(video_dir, config, output_file='results.json'):
    """Process videos sequentially (simple, lower memory)."""
    video_paths = list(Path(video_dir).glob('*.mp4'))
    
    print(f"Processing {len(video_paths)} videos sequentially...")
    results = []
    
    for video_path in video_paths:
        result = process_single_video(video_path, config)
        results.append(result)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def batch_process_parallel(video_dir, config, max_workers=4, output_file='results.json'):
    """Process videos in parallel (faster, more memory)."""
    video_paths = list(Path(video_dir).glob('*.mp4'))
    
    print(f"Processing {len(video_paths)} videos with {max_workers} workers...")
    
    # Partial function with fixed config
    process_func = partial(process_single_video, config=config)
    
    # Parallel execution
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_func, video_paths))
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def export_to_csv(results, csv_file='results.csv'):
    """Export results to CSV."""
    if not results:
        print("No results to export")
        return
    
    # Get all keys from first successful result
    successful = [r for r in results if r.get('success', False)]
    if not successful:
        print("No successful results to export")
        return
    
    fieldnames = successful[0].keys()
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in successful:
            writer.writerow(result)
    
    print(f"Results exported to: {csv_file}")

def print_summary(results):
    """Print summary statistics."""
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]
    
    if not successful:
        print("\nNo successful results")
        return
    
    # Calculate statistics
    h_values = [r['h_dfa'] for r in successful]
    d_values = [r['d_box'] for r in successful]
    fake_count = sum(1 for r in successful if r.get('likely_fake', False))
    
    print(f"\n{'='*60}")
    print(f"Batch Processing Summary")
    print(f"{'='*60}")
    print(f"Total videos:     {len(results)}")
    print(f"Successful:       {len(successful)}")
    print(f"Failed:           {len(failed)}")
    print(f"\nDeepfake Statistics:")
    print(f"Likely fake:      {fake_count} ({fake_count/len(successful)*100:.1f}%)")
    print(f"Likely real:      {len(successful)-fake_count} ({(len(successful)-fake_count)/len(successful)*100:.1f}%)")
    print(f"\nHurst Exponent (H):")
    print(f"Mean:             {sum(h_values)/len(h_values):.3f}")
    print(f"Min:              {min(h_values):.3f}")
    print(f"Max:              {max(h_values):.3f}")
    print(f"\nFractal Dimension (D):")
    print(f"Mean:             {sum(d_values)/len(d_values):.3f}")
    print(f"Min:              {min(d_values):.3f}")
    print(f"Max:              {max(d_values):.3f}")

def main():
    # Configuration
    video_dir = 'videos/'  # Directory containing videos
    config = ConfigPresets.production_fast()  # Fast preset
    parallel = True  # Use parallel processing
    max_workers = 4  # Number of parallel workers
    
    # Process videos
    if parallel:
        results = batch_process_parallel(video_dir, config, max_workers)
    else:
        results = batch_process_sequential(video_dir, config)
    
    # Print summary
    print_summary(results)
    
    # Export to CSV
    export_to_csv(results)

if __name__ == '__main__':
    main()
