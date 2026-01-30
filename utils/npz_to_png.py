import os
import numpy as np
import cv2
from PIL import Image, ExifTags


def npz_to_outputs(npz_path, original_img_path, output_dir, base_name):
    os.makedirs(output_dir, exist_ok=True)

    data = np.load(npz_path)
    print("🧠 NPZ KEYS:", data.files)

    original = cv2.imread(original_img_path)
    if original is None:
        raise RuntimeError("Original image not found")

    # ✅ AUTO-DETECT TruFor output key
    # ✅ AUTO-DETECT TruFor output key
    possible_keys = [
        "out",           # most common in TruFor
        "pred",
        "heatmap",
        "map",
        "tamper_map",
        "segmentation",
        "mask"
    ]

    heatmap = None
    for k in possible_keys:
        if k in data:
            heatmap = data[k]
            break

    if heatmap is None:
        raise RuntimeError(f"No known heatmap key found. Keys present: {data.files}")


    # -------- HEATMAP --------
    heatmap = heatmap.astype(np.float32)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = (heatmap * 255).astype(np.uint8)

    heatmap = cv2.resize(
        heatmap,
        (original.shape[1], original.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    heatmap_path = os.path.join(output_dir, f"{base_name}_heatmap.png")
    cv2.imwrite(heatmap_path, heatmap_img)

    # -------- OVERLAY --------
    overlay = cv2.addWeighted(original, 0.6, heatmap_img, 0.4, 0)
    overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
    cv2.imwrite(overlay_path, overlay)

    score = float(np.mean(heatmap) / 255.0)

    if score > 0.6:
        verdict = "Tampered"
        risk = "High"
    elif score > 0.3:
        verdict = "Suspicious"
        risk = "Medium"
    else:
        verdict = "Authentic"
        risk = "Low"
    
    metrics = compute_image_metrics(original)
    exif = extract_exif_anomalies(original_img_path)

    return {
        "heatmap": heatmap_path,
        "overlay": overlay_path,
        "score": score,
        "verdict": verdict,
        "risk": risk,
        # 🔹 NEW (added, not replacing)
        "metrics": metrics,
        "exif": exif
    }

def extract_exif_anomalies(image_path):
    anomalies = []

    try:
        img = Image.open(image_path)
        exif = img._getexif()

        if not exif:
            anomalies.append("Missing EXIF metadata")
            return anomalies

        exif_data = {
            ExifTags.TAGS.get(k, k): v for k, v in exif.items()
        }

        if "Software" in exif_data:
            if any(x in str(exif_data["Software"]).lower()
                   for x in ["photoshop", "gimp", "ai", "generator"]):
                anomalies.append("Editing software detected")

        if "DateTimeOriginal" not in exif_data:
            anomalies.append("Missing original capture timestamp")

        if "Make" not in exif_data or "Model" not in exif_data:
            anomalies.append("Camera make/model missing")

    except Exception:
        anomalies.append("EXIF parsing failed")

    return anomalies

def compute_image_metrics(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    noise = np.std(gray) / 255.0

    edges = cv2.Canny(gray, 80, 160)
    edge_density = np.mean(edges) / 255.0

    blockiness = np.std(gray[::8, :] - gray[1::8, :]) / 255.0
    ringing = np.mean(np.abs(cv2.Laplacian(gray, cv2.CV_32F))) / 255.0

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = np.mean(hsv[:, :, 1]) / 255.0

    return {
        "blur": round(blur, 1),
        "noise": round(noise, 3),
        "edge_density": round(edge_density, 3),
        "blockiness": round(blockiness, 3),
        "ringing": round(ringing, 3),
        "saturation": round(saturation, 3)
    }