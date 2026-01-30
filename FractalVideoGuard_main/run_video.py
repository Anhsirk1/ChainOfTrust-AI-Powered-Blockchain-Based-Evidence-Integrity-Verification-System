import subprocess
import os
import json
import math
import cv2
import numpy as np

ROOT = os.path.abspath(os.path.dirname(__file__))
PYTHON = os.path.join(ROOT, "venv_fg", "Scripts", "python.exe")
SCRIPT = os.path.join(ROOT, "fractalvideoguard_v0_5_2.py")


def sanitize_for_json(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]

    return obj


def run_video_analysis(video_path, output_json):

    video_path = os.path.abspath(video_path).replace("\\", "/")
    output_json = os.path.abspath(output_json)

    print(f"🎥 Running video analysis: {video_path}")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    # ▶ RUN FRACTALVIDEOGUARD
    result = subprocess.run(
        [
            PYTHON,
            SCRIPT,
            "--preset", "fast",
            "--extract",
            video_path
        ],
        cwd=ROOT,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FractalVideoGuard failed:\n{result.stderr}"
        )

    # ▶ PARSE JSON OUTPUT
    try:
        raw_data = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(
            "Invalid JSON output from FractalVideoGuard:\n" + result.stdout
        )

    # ▶ SANITIZE NaN / Inf
    clean_data = sanitize_for_json(raw_data)
        # ================= EXTRA FORENSIC OUTPUTS =================
    video_id = os.path.splitext(os.path.basename(video_path))[0]

    base_dir = os.path.dirname(output_json)

    frames_dir = os.path.join(base_dir, "frames")
    heatmap_dir = os.path.join(base_dir, "heatmaps")

    frames = extract_frames(video_path, frames_dir)

    heatmaps = []
    for f in frames:
        frame_path = os.path.join(frames_dir, f)
        img = cv2.imread(frame_path)

        hname = f.replace("frame", "heatmap")
        hpath = os.path.join(heatmap_dir, hname)

        os.makedirs(heatmap_dir, exist_ok=True)
        generate_heatmap(img, hpath)
        heatmaps.append(hname)

    # Risk timeline (simple & fast)
    timeline = []
    features = clean_data.get("features", {})   
    block = features.get("blockiness_mean", 0)
    ring = features.get("ringing_mean", 0)

    for i in range(len(frames)):
        score = round(min((block * 10 + ring) / 5, 10), 2)
        timeline.append({"frame": i, "risk": score})

    # Attach to JSON
    clean_data["frames"] = frames
    clean_data["heatmaps"] = heatmaps
    clean_data["timeline"] = timeline


    # ▶ SAVE JSON FILE
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, indent=2)

    return clean_data

def extract_frames(video_path, out_dir, count=6):
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        return []

    indices = np.linspace(0, total - 1, count, dtype=int)
    frames = []

    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            name = f"frame_{i}.jpg"
            cv2.imwrite(os.path.join(out_dir, name), frame)
            frames.append(name)

    cap.release()
    return frames

def generate_heatmap(frame, out_path):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    mag = np.abs(lap)

    # 🔥 Strong normalization
    mag = np.clip(mag * 4, 0, 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(mag, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)
    cv2.imwrite(out_path, overlay)