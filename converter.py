try:
    import torch
    from model_arch import RRDBNet
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import os
import requests

try:
    import openvino as ov
    import numpy as np
    HAS_OV = True
except ImportError:
    HAS_OV = False


def convert_model():
    OV_PATH = "weights/model.xml"
    # Switching to RealESRGAN_x4plus_anime_6B for 4x speed and significantly lower RAM
    WEIGHTS_PATH = "weights/RealESRGAN_x4plus_anime_6B.pth"
    MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"

    os.makedirs("weights", exist_ok=True)

    # If model already exists, we are done
    if os.path.exists(OV_PATH):
        print(f"[+] Intel OpenVINO model already exists at {OV_PATH}. Skipping conversion.")
        return

    if not HAS_TORCH or not HAS_OV:
        print("[!] Torch or OpenVINO not installed. Skipping conversion.")
        print("[!] Run conversion locally: pip install -r requirements-dev.txt")
        return

    # 1. Download PyTorch weights
    if not os.path.exists(WEIGHTS_PATH):
        print(f"[*] Downloading lightweight model ({MODEL_URL})...")
        response = requests.get(MODEL_URL)
        with open(WEIGHTS_PATH, "wb") as f:
            f.write(response.content)
        print("[+] Download successful.")

    # 2. Load PyTorch Model (6 blocks instead of 23)
    print("[*] Initializing Compact Neural Architecture...")
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    
    print("[*] Loading Weights...")
    loadnet = torch.load(WEIGHTS_PATH, map_location='cpu')
    
    keyname = 'params_ema' if 'params_ema' in loadnet else 'params'
    state_dict = loadnet[keyname]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('body.'): new_k = k.replace('body.', 'RRDB_trunk.')
        elif k.startswith('conv_body'): new_k = k.replace('conv_body', 'trunk_conv')
        elif k.startswith('conv_up1'): new_k = k.replace('conv_up1', 'upconv1')
        elif k.startswith('conv_up2'): new_k = k.replace('conv_up2', 'upconv2')
        elif k.startswith('conv_hr'): new_k = k.replace('conv_hr', 'HRconv')
        elif k.startswith('conv_last'): new_k = k.replace('conv_last', 'conv_last')
        else: new_k = k
        new_state_dict[new_k] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    # Explicitly clear torch cache and delete state dicts to free memory
    del loadnet, state_dict, new_state_dict
    gc.collect()

    # 3. Direct Conversion to OpenVINO (Skip ONNX for speed)
    print("[*] Transitioning to Intel Ultra-Insight Engine (Optimizing)...")
    # Wrap in example input for shape inference - using smaller size to save memory
    example_input = torch.randn(1, 3, 32, 32)
    ov_model = ov.convert_model(model, example_input=example_input)
    
    # Enable dynamic shapes for variable image sizes
    ov_model.reshape([-1, 3, -1, -1])
    
    print(f"[*] Compiling Intel optimized IR: {OV_PATH}")
    ov.save_model(ov_model, OV_PATH, compress_to_fp16=True)
    print("[+] Model files saved.")
    print("[+] Optimization Complete. Engine Ready.")

if __name__ == "__main__":
    convert_model()
