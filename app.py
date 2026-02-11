import cv2
import gc
import os
import time
import uuid
import threading
import numpy as np
import openvino as ov
from flask import Flask, request, render_template, jsonify, send_file
from flask.wrappers import Response
from flask_cors import CORS
from flask_compress import Compress
from io import BytesIO

app = Flask(__name__)
CORS(app)
Compress(app)

# OpenVINO Configuration
MODEL_XML = "weights/model.xml"
MODEL_BIN = "weights/model.bin"

# Initialize OpenVINO Core
core = ov.Core()
device = "CPU"
compiled_model = None

def get_compiled_model():
    global compiled_model
    if compiled_model is None:
        if not os.path.exists(MODEL_XML):
            print("OpenVINO model not found. Please run converter.py first.")
            return None
        model = core.read_model(MODEL_XML)
        compiled_model = core.compile_model(model, device)
    return compiled_model

# Global store for task progress
progress_store = {}

def apply_premium_filters(img):
    """
    Apply 'Ultra-Insight' post-processing filters for the stand-out effect.
    """
    # 1. Selective Texture Enhancement (CLAHE)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # 2. Intelligent Detail Restoration (Unsharp Masking)
    gaussian_3 = cv2.GaussianBlur(img, (0, 0), 2.0)
    img = cv2.addWeighted(img, 1.5, gaussian_3, -0.5, 0)
    
    # 3. Cinematic Color Adjustment
    img = cv2.convertScaleAbs(img, alpha=1.05, beta=2) # Subtle contrast and brightness
    
    return img

def enhance_image(img, task_id):
    """
    Enhance image using OpenVINO with tile processing.
    """
    try:
        model_infer = get_compiled_model()
        if model_infer is None:
            raise Exception("Inference engine not ready. Model conversion might still be in progress.")

        start_time = time.time()
        
        progress_store[task_id] = {
            'progress': 2,
            'remaining': -1,
            'status': 'starting',
            'message': 'Optimizing for Intel CPU...'
        }
        
        # Preprocessing
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        tile_size = 512 
        tile_pad = 10
        scale = 4

        b, c, h, w = img.shape
        img_HR = np.zeros((b, c, h * scale, w * scale), dtype=np.float32)
        
        h_idx_list = list(range(0, h, tile_size))
        w_idx_list = list(range(0, w, tile_size))
        total_tiles = len(h_idx_list) * len(w_idx_list)
        tiles_done = 0

        input_layer = model_infer.input(0)
        output_layer = model_infer.output(0)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                tiles_done += 1
                progress = int(((tiles_done - 1) / total_tiles) * 100)
                
                # Dynamic ETA calculation
                elapsed = time.time() - start_time
                avg_time_per_tile = elapsed / tiles_done
                remaining_tiles = total_tiles - tiles_done
                est_remaining = int(remaining_tiles * avg_time_per_tile)

                progress_store[task_id].update({
                    'progress': max(2, progress),
                    'status': 'processing',
                    'message': f"Analyzing Patterns (Tile {tiles_done}/{total_tiles})",
                    'remaining': est_remaining
                })
                
                h_start, h_end = h_idx, min(h_idx + tile_size, h)
                w_start, w_end = w_idx, min(w_idx + tile_size, w)
                
                h_start_pad = max(h_start - tile_pad, 0)
                h_end_pad = min(h_end + tile_pad, h)
                w_start_pad = max(w_start - tile_pad, 0)
                w_end_pad = min(w_end + tile_pad, w)

                input_tile = img[:, :, h_start_pad:h_end_pad, w_start_pad:w_end_pad]
                
                # OpenVINO Inference
                res = model_infer([input_tile])[output_layer]

                h_start_out = (h_start - h_start_pad) * scale
                h_end_out = (h_end - h_start_pad) * scale
                w_start_out = (w_start - w_start_pad) * scale
                w_end_out = (w_end - w_start_pad) * scale
                
                img_HR[:, :, h_start * scale:h_end * scale, w_start * scale:w_end * scale] = \
                    res[:, :, h_start_out:h_end_out, w_start_out:w_end_out]
                
                # Free memory after tile inference
                del res
                gc.collect()

        # Post-processing
        output = np.squeeze(img_HR).clip(0, 1)
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        
        # Apply Ultra-Insight premium filters
        progress_store[task_id].update({'message': 'Mastering Detail & Lighting...'})
        output = apply_premium_filters(output)
        
        _, encoded_img = cv2.imencode('.jpg', output, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        byte_image = encoded_img.tobytes()
        
        progress_store[task_id] = {
            'progress': 100,
            'remaining': 0,
            'status': 'completed',
            'message': 'Restoration Complete',
            'result': byte_image
        }
        
    except Exception as e:
        print(f"Error in background task {task_id}: {e}")
        progress_store[task_id] = {
            'status': 'error',
            'progress': 0,
            'message': f"Error: {str(e)}"
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
            
        file = request.files['image']
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        task_id = str(uuid.uuid4())
        progress_store[task_id] = {
            'progress': 0,
            'remaining': 0,
            'status': 'starting'
        }
        
        # Start background processing
        thread = threading.Thread(target=enhance_image, args=(img, task_id))
        thread.start()
        
        return jsonify({'task_id': task_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status/<task_id>')
def status(task_id):
    if task_id not in progress_store:
        return jsonify({'error': 'Task not found'}), 404
    
    # Return everything except the raw image data
    data = progress_store[task_id].copy()
    if 'result' in data:
        del data['result']
    return jsonify(data)

@app.route('/download/<task_id>')
def download(task_id):
    if task_id not in progress_store or 'result' not in progress_store[task_id]:
        return "Image not ready or task not found", 404
        
    return send_file(
        BytesIO(progress_store[task_id]['result']),
        mimetype='image/jpeg',
        as_attachment=True,
        download_name='enhanced_image.jpg'
    )

@app.route('/image/<task_id>')
def get_image(task_id):
    if task_id not in progress_store or 'result' not in progress_store[task_id]:
        return "Image not ready or task not found", 404
        
    return Response(progress_store[task_id]['result'], mimetype='image/jpeg')

if __name__ == '__main__':
    # Disable reloader to prevent high RAM/CPU usage when scanning model files
    app.run(debug=True, threaded=True, use_reloader=False)