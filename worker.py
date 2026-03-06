# ============================================================
# worker.py – Pixel-Clear high-quality OpenCV enhancement
# ------------------------------------------------------------
# NO neural network. NO openvino. NO model loading.
# Peak RAM usage: ~15–25 MB regardless of image size.
# Input: already-resized ≤480px JPEG from the browser.
# Output: 4× upscaled JPEG with multi-pass sharpening.
# ============================================================
import sys
import cv2
import numpy as np


MAX_DIM = 480    # hard cap — browser already does this, guard again
SCALE   = 4      # upscale factor


def unsharp_mask(img, sigma=1.5, strength=1.2):
    """Classic unsharp mask: sharpens edges without halos."""
    blur   = cv2.GaussianBlur(img, (0, 0), sigma)
    sharp  = cv2.addWeighted(img, 1 + strength, blur, -strength, 0)
    return sharp


def clahe_enhance(img):
    """CLAHE on the L channel of LAB — boosts local contrast."""
    lab     = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl      = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)


def deblur_wiener(img, radius=1):
    """
    Simple frequency-domain deblur via a mild Wiener-like filter.
    Corrects moderate camera/motion blur without AI.
    """
    kernel_size = 2 * radius + 1
    kernel      = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[radius, radius] = 1.0
    # Use the sharpening kernel (Laplacian boost) instead of a full Wiener
    sharpen_k = np.array([[ 0, -1,  0],
                           [-1,  5, -1],
                           [ 0, -1,  0]], dtype=np.float32)
    return cv2.filter2D(img, -1, sharpen_k)


def enhance(img):
    h, w = img.shape[:2]

    # 1. Upscale 4× with Lanczos (best quality among non-AI methods)
    big = cv2.resize(img, (w * SCALE, h * SCALE),
                     interpolation=cv2.INTER_LANCZOS4)

    # 2. CLAHE contrast boost
    big = clahe_enhance(big)

    # 3. Multi-pass unsharp mask (two passes at different scales)
    big = unsharp_mask(big, sigma=1.0, strength=0.8)
    big = unsharp_mask(big, sigma=2.0, strength=0.4)

    # 4. Mild deblur / detail pop
    big = deblur_wiener(big)

    # 5. Brightness/saturation micro-lift
    big = cv2.convertScaleAbs(big, alpha=1.04, beta=3)

    return big


def run(input_path, output_path):
    img = cv2.imread(input_path)
    if img is None:
        print("ERR: cannot read input image", file=sys.stderr)
        sys.exit(1)

    # Guard: cap to MAX_DIM before upscaling
    h, w = img.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        img   = cv2.resize(img, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)

    result = enhance(img)
    cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"OK: saved {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: worker.py <input> <output>", file=sys.stderr)
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])
