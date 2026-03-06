import sys
import os
import cv2
import numpy as np
import requests

# ------------------------------------------------------------------
# AI Model Configuration (FSRCNN 4x)
# This model is tiny (88 KB) but significantly better than Lanczos4.
# It runs with almost ZERO memory overhead compared to RealESRGAN.
# ------------------------------------------------------------------
MODEL_PATH = "weights/FSRCNN_x4.pb"
MODEL_URL  = "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x4.pb"
MAX_DIM    = 480    # hard cap — browser already does this, guard again
SCALE      = 4      # upscale factor

def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("weights", exist_ok=True)
        print(f"[*] Downloading Super-Res Model ({MODEL_URL})...")
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        print("[+] Download complete.")

def apply_filters(img):
    """Refined post-processing for AI output."""
    lab     = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # CLAHE for local contrast enhancement
    cl      = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    img     = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    
    # Mild sharpening (unsharp mask)
    blur    = cv2.GaussianBlur(img, (0, 0), 2.0)
    img     = cv2.addWeighted(img, 1.3, blur, -0.3, 0)
    
    # Mild brightness/contrast lift
    return cv2.convertScaleAbs(img, alpha=1.03, beta=2)

def run_ai_upscale(img):
    # Initialize OpenCV Super-Resolution module
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(MODEL_PATH)
    sr.setModel("fsrcnn", SCALE)
    
    # Upscale (one-shot, very fast on CPU)
    return sr.upsample(img)

def run(input_path, output_path):
    download_model()
    
    img = cv2.imread(input_path)
    if img is None:
        print("ERR: cannot read input image", file=sys.stderr)
        sys.exit(1)

    h, w = img.shape[:2]
    # Guard: cap to MAX_DIM before upscaling
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        img   = cv2.resize(img, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)

    # 1. AI Super-Resolution (FSRCNN)
    print("[*] Running AI Restoration Engine...")
    result = run_ai_upscale(img)
    
    # 2. Premium Post-processing
    print("[*] Perfecting output filters...")
    result = apply_filters(result)
    
    cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"OK: saved {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: worker.py <input> <output>", file=sys.stderr)
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])
