#!/usr/bin/env python3
"""
Basic usage example for FractalVideoGuard.

Demonstrates:
- Simple feature extraction
- Result interpretation
- Deepfake scoring
"""

from fractalvideoguard_v0_5_2 import extract_features, ConfigPresets

def main():
    # Video to analyze
    video_path = 'test_video.mp4'
    
    # Use fast preset for quick analysis
    config = ConfigPresets.production_fast()
    
    # Extract features
    print(f"Analyzing: {video_path}")
    features, debug = extract_features(video_path, config=config)
    
    # Key fractal features
    h = features['hurst_dfa']              # Hurst exponent (LRD)
    d = features['fractal_dim_box_mean']   # Fractal dimension
    r2 = features['hurst_dfa_r2']          # DFA fit quality
    
    # Frequency features
    dct_hf = features['dct_hf_mean']       # DCT high-frequency energy
    ringing = features['ringing_mean']     # Ringing artifacts
    
    # Display results
    print(f"\n{'='*50}")
    print(f"Fractal Features:")
    print(f"{'='*50}")
    print(f"Hurst Exponent (H):     {h:.3f}  (Real ≈ 0.70)")
    print(f"Fractal Dimension (D):  {d:.3f}  (Real ≈ 1.35)")
    print(f"DFA R²:                 {r2:.3f} (Quality: {'Good' if r2 > 0.90 else 'Poor'})")
    
    print(f"\n{'='*50}")
    print(f"Frequency Features:")
    print(f"{'='*50}")
    print(f"DCT High-Freq Energy:   {dct_hf:.3f}")
    print(f"Ringing Artifacts:      {ringing:.3f}")
    
    # Simple deepfake scoring
    h_deviation = abs(0.70 - h)
    d_deviation = abs(1.35 - d)
    score_fake = (h_deviation * 2.0) + (d_deviation * 1.5)
    
    print(f"\n{'='*50}")
    print(f"Assessment:")
    print(f"{'='*50}")
    print(f"Deviation from natural attractors: {score_fake:.3f}")
    
    if h < 0.60 or d < 1.20:
        print("⚠️  WARNING: Possible deepfake detected!")
        print(f"   - H = {h:.3f} is below natural range (0.65-0.75)")
        print(f"   - D = {d:.3f} is below natural range (1.25-1.45)")
    else:
        print("✅ Video appears authentic")
        print(f"   - H = {h:.3f} is within natural range")
        print(f"   - D = {d:.3f} is within natural range")
    
    # Metadata
    print(f"\n{'='*50}")
    print(f"Video Metadata:")
    print(f"{'='*50}")
    meta = debug['video_metadata']
    print(f"Source:     {meta['source']}")
    print(f"Native FPS: {meta.get('fps_native', 'N/A')}")
    print(f"Rotation:   {meta.get('rotation_deg', 0)}°")
    print(f"Version:    {debug['version']}")

if __name__ == '__main__':
    main()
