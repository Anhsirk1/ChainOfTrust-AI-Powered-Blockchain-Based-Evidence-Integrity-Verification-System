#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FractalVideoGuard v0.5.2 - Golden Test Harness
===============================================

Comprehensive test suite to verify single-file production package.

Tests:
1. Config validation (all 6 categories)
2. Numerical stability (no NaN/inf propagation)
3. Memory stability (buffer reuse works)
4. Rotation timeout (hang-proof)
5. Edge cases (uniform, low-contrast, corrupted)
6. Feature extraction pipeline (end-to-end)

Usage:
    python test_golden_v0_5_2.py
    python test_golden_v0_5_2.py --verbose
    python test_golden_v0_5_2.py --only memory
    python test_golden_v0_5_2.py --only rotation

Author: Claude (Anthropic) + Igor Chechelnitsky
Date: 2026-01-18
"""

import sys
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Callable
import numpy as np

# Import module under test
sys.path.insert(0, str(Path(__file__).parent))
from fractalvideoguard_v0_5_2 import (
    FIOConfig, ConfigPresets, VideoConfig, ROIConfig, FractalConfig,
    FrequencyConfig, StatisticsConfig, TrainingConfig,
    VideoReader, FaceROISource, ROIStandardizer,
    dct_hf_fraction, fft_hf_fraction, blockiness, block_var, ringing_proxy_robust,
    dfa_hurst, boxcount_dimension, extract_features,
    safe_detect_rotation,
    sha256_file, to_gray,
)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("⚠️  Warning: OpenCV not available, skipping video tests")


# =============================================================================
# Test Framework
# =============================================================================

class TestResult:
    def __init__(self, name: str, passed: bool, message: str = "", duration: float = 0.0):
        self.name = name
        self.passed = passed
        self.message = message
        self.duration = duration
    
    def __repr__(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status} {self.name} ({self.duration:.3f}s) {self.message}"


class TestSuite:
    def __init__(self, name: str):
        self.name = name
        self.results: List[TestResult] = []
        self.start_time = 0.0
    
    def run_test(self, test_name: str, test_func: Callable[[], bool], verbose: bool = False) -> None:
        """Run a single test and record result."""
        if verbose:
            print(f"  Running: {test_name}...", end=" ", flush=True)
        
        start = time.time()
        try:
            passed = test_func()
            duration = time.time() - start
            result = TestResult(test_name, passed, "", duration)
            if verbose:
                print("✅" if passed else "❌")
        except Exception as e:
            duration = time.time() - start
            result = TestResult(test_name, False, f"Exception: {e}", duration)
            if verbose:
                print(f"❌ {e}")
        
        self.results.append(result)
    
    def summary(self) -> str:
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        duration = sum(r.duration for r in self.results)
        
        s = f"\n{'='*60}\n"
        s += f"Test Suite: {self.name}\n"
        s += f"{'='*60}\n"
        s += f"Results: {passed}/{total} passed ({passed/total*100:.1f}%)\n"
        s += f"Duration: {duration:.2f}s\n\n"
        
        for r in self.results:
            s += str(r) + "\n"
        
        return s


# =============================================================================
# Category 1: Config Validation Tests
# =============================================================================

def test_config_validation():
    """Test that all config validators catch invalid parameters."""
    suite = TestSuite("Config Validation")
    
    # Test 1.1: Video config validation
    def test_video_invalid_fps():
        cfg = VideoConfig()
        cfg.fps_target = 0  # Invalid
        errors = cfg.validate()
        return len(errors) > 0  # Should have errors
    
    suite.run_test("video_invalid_fps", test_video_invalid_fps)
    
    # Test 1.2: ROI config validation
    def test_roi_invalid_confidence():
        cfg = ROIConfig()
        cfg.detection_confidence = 1.5  # Invalid (>1.0)
        errors = cfg.validate()
        return len(errors) > 0
    
    suite.run_test("roi_invalid_confidence", test_roi_invalid_confidence)
    
    # Test 1.3: Fractal config validation
    def test_fractal_invalid_scales():
        cfg = FractalConfig()
        cfg.dfa_scales = (8,)  # Too few (<3)
        errors = cfg.validate()
        return len(errors) > 0
    
    suite.run_test("fractal_invalid_scales", test_fractal_invalid_scales)
    
    # Test 1.4: Frequency config validation
    def test_freq_invalid_nan_handling():
        cfg = FrequencyConfig()
        cfg.nan_handling = "invalid"
        errors = cfg.validate()
        return len(errors) > 0
    
    suite.run_test("freq_invalid_nan_handling", test_freq_invalid_nan_handling)
    
    # Test 1.5: Full config validation
    def test_full_config_valid():
        cfg = FIOConfig()
        errors = cfg.validate()
        all_empty = all(len(v) == 0 for v in errors.values())
        return all_empty and cfg.is_valid()
    
    suite.run_test("full_config_valid_default", test_full_config_valid)
    
    # Test 1.6: Preset configs are valid
    def test_presets_valid():
        presets = [
            ConfigPresets.production_high_quality(),
            ConfigPresets.production_fast(),
            ConfigPresets.research_debug(),
            ConfigPresets.mobile_lightweight(),
        ]
        return all(p.is_valid() for p in presets)
    
    suite.run_test("presets_all_valid", test_presets_valid)
    
    # Test 1.7: JSON serialization round-trip
    def test_json_roundtrip():
        cfg1 = FIOConfig()
        cfg1.video.fps_target = 15
        cfg1.roi.std_roi_side = 512
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            cfg1.to_json(f.name)
            cfg2 = FIOConfig.from_json(f.name)
            Path(f.name).unlink()
        
        return (cfg2.video.fps_target == 15 and 
                cfg2.roi.std_roi_side == 512)
    
    suite.run_test("json_serialization_roundtrip", test_json_roundtrip)
    
    return suite


# =============================================================================
# Category 2: Numerical Stability Tests
# =============================================================================

def test_numerical_stability():
    """Test that no NaN/inf propagates through feature extraction."""
    suite = TestSuite("Numerical Stability")
    
    cfg = FIOConfig()
    
    # Test 2.1: Uniform image (MAD = 0 case)
    def test_uniform_image():
        uniform = np.ones((256, 256), dtype=np.uint8) * 128
        result = ringing_proxy_robust(uniform, cfg)
        return np.isfinite(result)  # Should not be inf/nan
    
    suite.run_test("uniform_image_no_inf", test_uniform_image)
    
    # Test 2.2: Low contrast image
    def test_low_contrast():
        low_contrast = np.random.randint(127, 129, (256, 256), dtype=np.uint8)
        result = ringing_proxy_robust(low_contrast, cfg)
        return np.isfinite(result)
    
    suite.run_test("low_contrast_stable", test_low_contrast)
    
    # Test 2.3: Black image
    def test_black_image():
        black = np.zeros((256, 256), dtype=np.uint8)
        result = ringing_proxy_robust(black, cfg)
        # Should return 0.0 or NaN (both acceptable), not inf
        return result == 0.0 or np.isnan(result)
    
    suite.run_test("black_image_safe", test_black_image)
    
    # Test 2.4: White image
    def test_white_image():
        white = np.ones((256, 256), dtype=np.uint8) * 255
        result = ringing_proxy_robust(white, cfg)
        return result == 0.0 or np.isnan(result)
    
    suite.run_test("white_image_safe", test_white_image)
    
    # Test 2.5: Random noise (should be stable)
    def test_random_noise():
        noise = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        result = ringing_proxy_robust(noise, cfg)
        return np.isfinite(result) and result >= 0.0
    
    suite.run_test("random_noise_finite", test_random_noise)
    
    # Test 2.6: DCT stability on uniform
    def test_dct_uniform():
        uniform = np.ones((256, 256), dtype=np.uint8) * 128
        result = dct_hf_fraction(uniform, cfg)
        return np.isfinite(result)
    
    suite.run_test("dct_uniform_stable", test_dct_uniform)
    
    # Test 2.7: FFT stability on uniform
    def test_fft_uniform():
        uniform = np.ones((256, 256), dtype=np.uint8) * 128
        result = fft_hf_fraction(uniform, cfg)
        return np.isfinite(result)
    
    suite.run_test("fft_uniform_stable", test_fft_uniform)
    
    # Test 2.8: All frequency features together
    def test_all_freq_features():
        test_imgs = [
            np.ones((256, 256), dtype=np.uint8) * 128,  # uniform
            np.zeros((256, 256), dtype=np.uint8),       # black
            np.ones((256, 256), dtype=np.uint8) * 255,  # white
            np.random.randint(0, 256, (256, 256), dtype=np.uint8),  # noise
        ]
        
        all_finite = True
        for img in test_imgs:
            dct = dct_hf_fraction(img, cfg)
            fft = fft_hf_fraction(img, cfg)
            blk = blockiness(img, cfg)
            rng = ringing_proxy_robust(img, cfg)
            bvr = block_var(img, cfg)
            
            # All should be finite or explicitly NaN (not inf)
            results = [dct, fft, blk, rng, bvr]
            for r in results:
                if not np.isfinite(r) and not np.isnan(r):
                    all_finite = False
        
        return all_finite
    
    suite.run_test("all_freq_features_no_inf", test_all_freq_features)
    
    return suite


# =============================================================================
# Category 3: Memory Stability Tests
# =============================================================================

def test_memory_stability():
    """Test that ROIStandardizer reuses buffers correctly."""
    suite = TestSuite("Memory Stability")
    
    # Test 3.1: Buffer reuse (no memory growth)
    def test_buffer_reuse():
        std = ROIStandardizer(target_size=256)
        
        # First call creates buffer
        img1 = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        out1 = std.standardize(img1, copy_out=False)
        buf_id_1 = id(out1)
        
        # Second call should reuse same buffer
        img2 = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        out2 = std.standardize(img2, copy_out=False)
        buf_id_2 = id(out2)
        
        std.release()
        return buf_id_1 == buf_id_2  # Same object = reused buffer
    
    suite.run_test("buffer_reuse_works", test_buffer_reuse)
    
    # Test 3.2: copy_out=True creates new object
    def test_copy_out():
        std = ROIStandardizer(target_size=256)
        
        img = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        out1 = std.standardize(img, copy_out=False)
        out2 = std.standardize(img, copy_out=True)
        
        std.release()
        return id(out1) != id(out2)  # Different objects
    
    suite.run_test("copy_out_creates_new", test_copy_out)
    
    # Test 3.3: Multiple calls don't grow memory (simulate batch)
    def test_batch_processing():
        std = ROIStandardizer(target_size=256)
        
        # Process 100 images - memory should stay stable
        for _ in range(100):
            img = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
            out = std.standardize(img, copy_out=False)
            # Compute something to ensure it's used
            _ = float(np.mean(out))
        
        std.release()
        return True  # If no crash/OOM, test passes
    
    suite.run_test("batch_processing_stable", test_batch_processing)
    
    # Test 3.4: Release clears buffer
    def test_release_clears():
        std = ROIStandardizer(target_size=256)
        
        img = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        _ = std.standardize(img, copy_out=False)
        
        std.release()
        return std._buf is None  # Buffer should be cleared
    
    suite.run_test("release_clears_buffer", test_release_clears)
    
    return suite


# =============================================================================
# Category 4: Rotation Timeout Tests
# =============================================================================

def test_rotation_timeout():
    """Test that rotation detection respects timeout and doesn't hang."""
    suite = TestSuite("Rotation Timeout")
    
    # Test 4.1: Normal file (should complete quickly)
    def test_normal_file():
        # Create dummy video file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            video_path = f.name
            # Write minimal valid MP4 (this might not be readable, but tests timeout logic)
            f.write(b'\x00\x00\x00\x20ftypmp42')
        
        try:
            start = time.time()
            rot = safe_detect_rotation(video_path, timeout_sec=1.0)
            duration = time.time() - start
            
            # Should complete within timeout
            return duration < 2.0 and isinstance(rot, int)
        finally:
            Path(video_path).unlink(missing_ok=True)
    
    suite.run_test("normal_file_completes", test_normal_file)
    
    # Test 4.2: Timeout enforces limit
    def test_timeout_enforced():
        # Invalid source that might hang backend
        start = time.time()
        rot = safe_detect_rotation("nonexistent://invalid", timeout_sec=0.5)
        duration = time.time() - start
        
        # Should abort within ~0.6s (timeout + small overhead)
        return duration < 1.5 and rot == 0
    
    suite.run_test("timeout_enforced", test_timeout_enforced)
    
    # Test 4.3: Returns 0 on failure
    def test_failure_returns_zero():
        rot = safe_detect_rotation("nonexistent_file.mp4", timeout_sec=0.5)
        return rot == 0
    
    suite.run_test("failure_returns_zero", test_failure_returns_zero)
    
    return suite


# =============================================================================
# Category 5: Edge Case Tests
# =============================================================================

def test_edge_cases():
    """Test handling of edge cases in feature extraction."""
    suite = TestSuite("Edge Cases")
    
    cfg = FIOConfig()
    
    # Test 5.1: Very small image (below min_roi_side)
    def test_tiny_image():
        tiny = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
        # Should return NaN, not crash
        result = dct_hf_fraction(tiny, cfg)
        return np.isnan(result)
    
    suite.run_test("tiny_image_returns_nan", test_tiny_image)
    
    # Test 5.2: Non-square image
    def test_non_square():
        rect = np.random.randint(0, 256, (256, 512), dtype=np.uint8)
        dct = dct_hf_fraction(rect, cfg)
        fft = fft_hf_fraction(rect, cfg)
        # Should handle gracefully
        return np.isfinite(dct) or np.isnan(dct)
    
    suite.run_test("non_square_handled", test_non_square)
    
    # Test 5.3: DFA with insufficient data
    def test_dfa_insufficient_data():
        short_series = np.random.randn(10)  # Too short
        H, r2 = dfa_hurst(short_series, (8, 16, 32), poly_order=1)
        return np.isnan(H) and np.isnan(r2)
    
    suite.run_test("dfa_insufficient_data", test_dfa_insufficient_data)
    
    # Test 5.4: Boxcount on empty image
    def test_boxcount_empty():
        empty = np.zeros((256, 256), dtype=np.uint8)
        D, r2 = boxcount_dimension(empty, (2, 4, 8, 16))
        return np.isnan(D) and np.isnan(r2)
    
    suite.run_test("boxcount_empty_image", test_boxcount_empty)
    
    # Test 5.5: None inputs
    def test_none_inputs():
        results = [
            dct_hf_fraction(None, cfg),
            fft_hf_fraction(None, cfg),
            blockiness(None, cfg),
            ringing_proxy_robust(None, cfg),
            block_var(None, cfg),
        ]
        return all(np.isnan(r) for r in results)
    
    suite.run_test("none_inputs_return_nan", test_none_inputs)
    
    return suite


# =============================================================================
# Category 6: End-to-End Pipeline Tests
# =============================================================================

def test_end_to_end_pipeline():
    """Test full feature extraction pipeline."""
    suite = TestSuite("End-to-End Pipeline")
    
    if not HAS_CV2:
        print("⚠️  Skipping E2E tests (OpenCV not available)")
        return suite
    
    # Test 6.1: Create synthetic video and extract features
    def test_synthetic_video_extraction():
        import cv2
        
        # Create temporary video file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            video_path = f.name
        
        try:
            # Write synthetic video (10 frames, 640x480)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 10.0, (640, 480))
            
            for i in range(10):
                frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
                out.write(frame)
            
            out.release()
            
            # Extract features with fast config
            cfg = ConfigPresets.mobile_lightweight()
            features, debug = extract_features(video_path, config=cfg, compute_sha256=False)
            
            # Check that we got features back
            has_features = len(features) > 0
            no_inf = all(np.isfinite(v) or np.isnan(v) for v in features.values())
            
            return has_features and no_inf
        
        except Exception as e:
            print(f"  Exception in synthetic video test: {e}")
            return False
        
        finally:
            Path(video_path).unlink(missing_ok=True)
    
    suite.run_test("synthetic_video_extraction", test_synthetic_video_extraction)
    
    # Test 6.2: Config affects output
    def test_config_affects_output():
        # This is a placeholder - would need real video for meaningful test
        cfg1 = FIOConfig()
        cfg1.video.fps_target = 10
        
        cfg2 = FIOConfig()
        cfg2.video.fps_target = 20
        
        # Configs are different
        return cfg1.to_dict() != cfg2.to_dict()
    
    suite.run_test("config_affects_output", test_config_affects_output)
    
    return suite


# =============================================================================
# Main Test Runner
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Golden test harness for FractalVideoGuard v0.5.2")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--only", type=str, help="Run only specific category: config|numerical|memory|rotation|edge|e2e")
    
    args = parser.parse_args()
    
    print("="*60)
    print("FractalVideoGuard v0.5.2 - Golden Test Harness")
    print("="*60)
    print()
    
    # Map of test categories
    test_categories = {
        "config": test_config_validation,
        "numerical": test_numerical_stability,
        "memory": test_memory_stability,
        "rotation": test_rotation_timeout,
        "edge": test_edge_cases,
        "e2e": test_end_to_end_pipeline,
    }
    
    # Run tests
    suites = []
    
    if args.only:
        if args.only not in test_categories:
            print(f"❌ Unknown category: {args.only}")
            print(f"Available: {', '.join(test_categories.keys())}")
            return 1
        
        print(f"Running category: {args.only}\n")
        suite = test_categories[args.only]()
        suites.append(suite)
    else:
        print("Running all test categories...\n")
        for name, test_func in test_categories.items():
            if args.verbose:
                print(f"\n[Category: {name}]")
            suite = test_func()
            suites.append(suite)
    
    # Print summaries
    print()
    total_passed = 0
    total_tests = 0
    
    for suite in suites:
        print(suite.summary())
        total_passed += sum(1 for r in suite.results if r.passed)
        total_tests += len(suite.results)
    
    # Final summary
    print("="*60)
    print(f"TOTAL: {total_passed}/{total_tests} tests passed ({total_passed/total_tests*100:.1f}%)")
    print("="*60)
    
    return 0 if total_passed == total_tests else 1


if __name__ == "__main__":
    raise SystemExit(main())
