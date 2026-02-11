import torch
import os
import requests
from model_arch import RRDBNet
import openvino as ov
import numpy as np
import gc

def convert_model():
    MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    WEIGHTS_PATH = "weights/RealESRGAN_x4plus.pth"
    OV_PATH = "weights/model.xml"

    os.makedirs("weights", exist_ok=True)

    # 1. Download PyTorch weights
    if not os.path.exists(WEIGHTS_PATH):
        print(f"[*] Downloading source model ({MODEL_URL})...")
        response = requests.get(MODEL_URL)
        with open(WEIGHTS_PATH, "wb") as f:
            f.write(response.content)
        print("[+] Download successful.")

    # 2. Load PyTorch Model
    print("[*] Initializing Neural Architecture...")
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    
    print("[*] Loading Weights into Memory...")
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
    
    print("[*] Compiling Intel optimized IR: {OV_PATH}")
    ov.save_model(ov_model, OV_PATH, compress_to_fp16=True)
    print("[+] Model files saved.")
    print("[+] Optimization Complete. Engine Ready.")

if __name__ == "__main__":
    convert_model()
