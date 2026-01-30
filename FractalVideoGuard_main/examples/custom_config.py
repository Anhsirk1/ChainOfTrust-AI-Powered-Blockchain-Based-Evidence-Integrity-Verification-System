#!/usr/bin/env python3
"""
Custom configuration example for FractalVideoGuard.

Demonstrates:
- Creating custom configurations
- Parameter tuning
- Config export/import
- Environment variable overrides
"""

from fractalvideoguard_v0_5_2 import FIOConfig, ConfigPresets, extract_features
from pathlib import Path

def create_forensic_config():
    """Create configuration optimized for forensic analysis."""
    config = FIOConfig()
    
    # Video processing: Maximum temporal detail
    config.video.fps_target = 20              # High frame sampling
    config.video.max_frames = 1500            # Long videos
    config.video.rotation_timeout_sec = 5.0   # Patient timeout
    
    # ROI extraction: Best quality
    config.roi.std_roi_side = 512             # Large ROI for detail
    config.roi.use_mediapipe = True           # Best face detection
    config.roi.detection_confidence = 0.6     # Higher threshold
    config.roi.bbox_smoothing_alpha = 0.70    # More temporal smoothing
    config.roi.blur_threshold = 120.0         # Stricter quality
    
    # Fractal analysis: Maximum precision
    config.fractal.dfa_scales = (8, 16, 32, 64, 128, 256, 512)  # More scales
    config.fractal.dfa_min_rsquared = 0.98    # Very strict fit
    config.fractal.dfa_poly_order = 2         # Quadratic detrending
    config.fractal.boxcount_scales = (2, 4, 8, 16, 32, 64, 128)
    config.fractal.boxcount_min_rsquared = 0.95
    
    # Frequency analysis: Detailed sampling
    config.frequency.sample_rate_frames = 2   # Process most frames
    config.frequency.dct_block_size = 8       # Standard JPEG blocks
    config.frequency.fft_hf_ratio = 0.25      # High-frequency focus
    
    # Statistics: Maximum validation
    config.statistics.enable_bootstrap_ci = True
    config.statistics.bootstrap_n_samples = 1000
    config.statistics.bootstrap_confidence = 0.99  # 99% CI
    config.statistics.enable_surrogate_test = True
    config.statistics.surrogate_n_samples = 500
    config.statistics.random_seed = 2026      # Reproducibility
    
    return config

def create_mobile_config():
    """Create configuration optimized for mobile devices."""
    config = FIOConfig()
    
    # Video processing: Minimal sampling
    config.video.fps_target = 4               # Very sparse
    config.video.max_frames = 120             # Short videos
    config.video.rotation_timeout_sec = 1.0   # Quick timeout
    
    # ROI extraction: Fast methods
    config.roi.std_roi_side = 128             # Small ROI
    config.roi.use_mediapipe = False          # CPU-only
    config.roi.detection_confidence = 0.4     # Lower threshold
    config.roi.bbox_smoothing_alpha = 0.5     # Less smoothing
    
    # Fractal analysis: Minimal scales
    config.fractal.dfa_scales = (8, 16, 32, 64)  # Fewer scales
    config.fractal.dfa_min_rsquared = 0.85    # Relaxed threshold
    config.fractal.boxcount_scales = (2, 4, 8, 16)
    
    # Frequency analysis: Sparse sampling
    config.frequency.sample_rate_frames = 20  # Very sparse
    
    # Statistics: Disabled
    config.statistics.enable_bootstrap_ci = False
    config.statistics.enable_surrogate_test = False
    
    return config

def create_research_config():
    """Create configuration for academic research."""
    config = FIOConfig()
    
    # Maximum temporal resolution
    config.video.fps_target = 25
    config.video.max_frames = 2000
    
    # Multiple scales for multifractal analysis
    config.fractal.dfa_scales = tuple(2**i for i in range(3, 10))  # 8 to 512
    config.fractal.dfa_poly_order = 3         # Cubic detrending
    
    # Full statistical validation
    config.statistics.enable_bootstrap_ci = True
    config.statistics.bootstrap_n_samples = 2000
    config.statistics.enable_surrogate_test = True
    config.statistics.surrogate_n_samples = 1000
    
    return config

def save_and_load_config():
    """Demonstrate config serialization."""
    # Create custom config
    config = create_forensic_config()
    
    # Save to JSON
    config_file = 'forensic_config.json'
    config.to_json(config_file)
    print(f"Config saved to: {config_file}")
    
    # Load from JSON
    config2 = FIOConfig.from_json(config_file)
    print(f"Config loaded from: {config_file}")
    
    # Verify
    assert config2.video.fps_target == 20
    assert config2.roi.std_roi_side == 512
    print("✅ Config serialization works correctly")
    
    # Validate
    errors = config2.validate()
    if config2.is_valid():
        print("✅ Config is valid")
    else:
        print("❌ Config validation failed:")
        for category, errs in errors.items():
            if errs:
                print(f"  [{category}] {errs}")

def compare_configs():
    """Compare different configurations."""
    configs = {
        'forensic': create_forensic_config(),
        'mobile': create_mobile_config(),
        'research': create_research_config(),
    }
    
    print("\nConfiguration Comparison:")
    print(f"{'='*70}")
    print(f"{'Parameter':<30} | {'Forensic':>12} | {'Mobile':>12} | {'Research':>12}")
    print(f"{'='*70}")
    
    # Video parameters
    for name in ['fps_target', 'max_frames']:
        forensic = getattr(configs['forensic'].video, name)
        mobile = getattr(configs['mobile'].video, name)
        research = getattr(configs['research'].video, name)
        print(f"video.{name:<23} | {forensic:>12} | {mobile:>12} | {research:>12}")
    
    # ROI parameters
    for name in ['std_roi_side', 'use_mediapipe']:
        forensic = getattr(configs['forensic'].roi, name)
        mobile = getattr(configs['mobile'].roi, name)
        research = getattr(configs['research'].roi, name)
        print(f"roi.{name:<25} | {forensic!s:>12} | {mobile!s:>12} | {research!s:>12}")
    
    # Fractal parameters
    forensic_scales = len(configs['forensic'].fractal.dfa_scales)
    mobile_scales = len(configs['mobile'].fractal.dfa_scales)
    research_scales = len(configs['research'].fractal.dfa_scales)
    print(f"fractal.dfa_scales (count)     | {forensic_scales:>12} | {mobile_scales:>12} | {research_scales:>12}")
    
    # Statistics
    for name in ['enable_bootstrap_ci', 'enable_surrogate_test']:
        forensic = getattr(configs['forensic'].statistics, name)
        mobile = getattr(configs['mobile'].statistics, name)
        research = getattr(configs['research'].statistics, name)
        print(f"statistics.{name:<18} | {forensic!s:>12} | {mobile!s:>12} | {research!s:>12}")
    
    print(f"{'='*70}")

def environment_override_example():
    """Demonstrate environment variable overrides."""
    import os
    
    # Set environment variables
    os.environ['FIO_USE_ENV'] = '1'
    os.environ['FIO_VIDEO_FPS_TARGET'] = '15'
    os.environ['FIO_ROI_STD_ROI_SIDE'] = '512'
    os.environ['FIO_FRACTAL_DFA_MIN_RSQUARED'] = '0.95'
    
    # Load config with environment overrides
    config = FIOConfig.from_env(prefix='FIO_')
    
    print("\nEnvironment Variable Overrides:")
    print(f"FIO_VIDEO_FPS_TARGET      → {config.video.fps_target}")
    print(f"FIO_ROI_STD_ROI_SIDE      → {config.roi.std_roi_side}")
    print(f"FIO_FRACTAL_DFA_MIN_RSQUARED → {config.fractal.dfa_min_rsquared}")
    
    # Clean up
    for key in ['FIO_USE_ENV', 'FIO_VIDEO_FPS_TARGET', 'FIO_ROI_STD_ROI_SIDE', 'FIO_FRACTAL_DFA_MIN_RSQUARED']:
        os.environ.pop(key, None)

def main():
    print("FractalVideoGuard - Custom Configuration Examples\n")
    
    # 1. Compare configurations
    compare_configs()
    
    # 2. Save and load
    print("\n")
    save_and_load_config()
    
    # 3. Environment overrides
    print("\n")
    environment_override_example()
    
    # 4. Use custom config for analysis
    print("\n")
    print("Example: Using forensic config for analysis")
    config = create_forensic_config()
    
    # Uncomment to run actual analysis:
    # video_path = 'test_video.mp4'
    # features, debug = extract_features(video_path, config=config)
    # print(f"Hurst: {features['hurst_dfa']:.3f}")
    # print(f"Dimension: {features['fractal_dim_box_mean']:.3f}")

if __name__ == '__main__':
    main()
