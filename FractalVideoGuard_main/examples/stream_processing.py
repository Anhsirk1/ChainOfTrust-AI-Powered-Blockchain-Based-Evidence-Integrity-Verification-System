#!/usr/bin/env python3
"""
Stream processing example for FractalVideoGuard.

Demonstrates:
- RTSP camera stream analysis
- Webcam monitoring
- Continuous analysis loop
- Alert system
"""

import time
from datetime import datetime
from fractalvideoguard_v0_5_2 import extract_features, ConfigPresets

def analyze_stream(stream_url, config, check_interval=60):
    """
    Continuously analyze video stream.
    
    Args:
        stream_url: RTSP URL, HTTP stream, or webcam device ID (0, 1, etc.)
        config: FIOConfig object
        check_interval: Seconds between analyses
    """
    print(f"Monitoring stream: {stream_url}")
    print(f"Check interval: {check_interval}s")
    print(f"Press Ctrl+C to stop\n")
    
    iteration = 0
    
    try:
        while True:
            iteration += 1
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"[{timestamp}] Analysis #{iteration}...", end=' ', flush=True)
            
            try:
                # Extract features from stream
                features, debug = extract_features(stream_url, config=config)
                
                # Key indicators
                h = features['hurst_dfa']
                d = features['fractal_dim_box_mean']
                r2 = features['hurst_dfa_r2']
                
                # Check for anomalies
                h_anomaly = h < 0.60 or h > 0.80
                d_anomaly = d < 1.20 or d > 1.50
                quality_poor = r2 < 0.85
                
                if h_anomaly or d_anomaly:
                    print(f"⚠️  ALERT!")
                    print(f"   Hurst:    {h:.3f} {'(ANOMALY)' if h_anomaly else ''}")
                    print(f"   Dimension: {d:.3f} {'(ANOMALY)' if d_anomaly else ''}")
                    print(f"   Quality:   {r2:.3f} {'(POOR)' if quality_poor else ''}")
                    
                    # Send alert (implement your alert system here)
                    send_alert(timestamp, h, d, r2)
                else:
                    print(f"✅ Normal (H={h:.3f}, D={d:.3f}, R²={r2:.3f})")
                
            except Exception as e:
                print(f"❌ Error: {e}")
            
            # Wait before next check
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")

def send_alert(timestamp, h, d, r2):
    """
    Send alert when anomaly detected.
    
    Implement your alert system here:
    - Email notification
    - Webhook (Slack, Discord, etc.)
    - SMS
    - Database logging
    """
    # Example: Print to console
    print(f"\n{'='*60}")
    print(f"ALERT: Anomaly detected at {timestamp}")
    print(f"{'='*60}")
    print(f"Hurst Exponent:     {h:.3f} (expected: 0.65-0.75)")
    print(f"Fractal Dimension:  {d:.3f} (expected: 1.30-1.40)")
    print(f"DFA Quality:        {r2:.3f}")
    print(f"{'='*60}\n")
    
    # TODO: Implement actual alert system
    # Example: Send email
    # send_email(subject=f"Deepfake Alert {timestamp}", 
    #            body=f"H={h:.3f}, D={d:.3f}")
    
    # Example: POST to webhook
    # requests.post("https://hooks.slack.com/...", 
    #               json={"text": f"Anomaly: H={h:.3f}, D={d:.3f}"})

def monitor_webcam(device=0, duration_seconds=300):
    """
    Monitor webcam for specified duration.
    
    Args:
        device: Webcam device ID (0=default, 1=second camera, etc.)
        duration_seconds: How long to capture (seconds)
    """
    print(f"Monitoring webcam (device {device}) for {duration_seconds}s...")
    
    config = ConfigPresets.mobile_lightweight()
    
    # Adjust for quick webcam analysis
    config.video.max_frames = 120  # ~20 seconds @ 6 fps
    
    features, debug = extract_features(device, config=config)
    
    h = features['hurst_dfa']
    d = features['fractal_dim_box_mean']
    
    print(f"\nWebcam Analysis Results:")
    print(f"Hurst Exponent:    {h:.3f}")
    print(f"Fractal Dimension: {d:.3f}")
    
    if h > 0.60 and d > 1.20:
        print("✅ Webcam feed appears authentic")
    else:
        print("⚠️  Webcam feed may be manipulated or spoofed")
        print("   (Could also indicate low quality or compression)")
    
    return {'h': h, 'd': d, 'authentic': h > 0.60 and d > 1.20}

def main():
    # Choose monitoring mode
    mode = 'rtsp'  # Options: 'rtsp', 'webcam', 'http'
    
    if mode == 'rtsp':
        # RTSP camera stream
        stream_url = 'rtsp://192.168.1.100:554/stream'
        config = ConfigPresets.production_fast()
        analyze_stream(stream_url, config, check_interval=60)
        
    elif mode == 'webcam':
        # Webcam (device 0)
        monitor_webcam(device=0, duration_seconds=300)
        
    elif mode == 'http':
        # HTTP/HLS stream
        stream_url = 'http://example.com/live.m3u8'
        config = ConfigPresets.production_fast()
        analyze_stream(stream_url, config, check_interval=60)

if __name__ == '__main__':
    main()
