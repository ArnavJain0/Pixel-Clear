import cv2
import gc
import os
import time
import uuid
import threading
import sqlite3
import numpy as np
import openvino as ov
from flask import Flask, request, render_template, jsonify, send_file
from flask_cors import CORS
from flask_compress import Compress

app = Flask(__name__)
CORS(app)
Compress(app)

# Configuration
MODEL_XML = "weights/model.xml"
DB_PATH = "tasks.db"
RESULTS_DIR = "static/results"

# Smart Caching Globals
cached_model = None
last_activity_time = time.time()
cache_lock = threading.Lock()

os.makedirs(RESULTS_DIR, exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH, timeout=20)
    c = conn.cursor()
    c.execute('PRAGMA journal_mode=WAL')
    c.execute('''CREATE TABLE IF NOT EXISTS tasks
                 (id TEXT PRIMARY KEY, status TEXT, progress INTEGER, remaining INTEGER, 
                  message TEXT, result_path TEXT, timestamp REAL)''')
    conn.commit()
    conn.close()

init_db()

def update_task(task_id, status=None, progress=None, remaining=None, message=None, result_path=None):
    conn = sqlite3.connect(DB_PATH, timeout=20)
    c = conn.cursor()
    c.execute('PRAGMA journal_mode=WAL')
    updates = []
    params = []
    if status is not None:
        updates.append("status=?")
        params.append(status)
    if progress is not None:
        updates.append("progress=?")
        params.append(progress)
    if remaining is not None:
        updates.append("remaining=?")
        params.append(remaining)
    if message is not None:
        updates.append("message=?")
        params.append(message)
    if result_path is not None:
        updates.append("result_path=?")
        params.append(result_path)
    
    if updates:
        params.append(task_id)
        c.execute(f"UPDATE tasks SET {', '.join(updates)} WHERE id=?", params)
        conn.commit()
    conn.close()

def get_task(task_id):
    conn = sqlite3.connect(DB_PATH, timeout=20)
    c = conn.cursor()
    c.execute('PRAGMA journal_mode=WAL')
    c.execute("SELECT status, progress, remaining, message, result_path FROM tasks WHERE id=?", (task_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return {
            'status': row[0],
            'progress': row[1],
            'remaining': row[2],
            'message': row[3],
            'result_path': row[4]
        }
    return None

def cleanup_old_tasks():
    now = time.time()
    conn = sqlite3.connect(DB_PATH, timeout=20)
    c = conn.cursor()
    c.execute('PRAGMA journal_mode=WAL')
    c.execute("SELECT result_path FROM tasks WHERE timestamp < ?", (now - 3600,))
    rows = c.fetchall()
    for row in rows:
        if row[0] and os.path.exists(row[0]):
            try: os.remove(row[0])
            except: pass
    c.execute("DELETE FROM tasks WHERE timestamp < ?", (now - 3600,))
    conn.commit()
    conn.close()

def cache_watchdog():
    """Purge cached model after 5 minutes of inactivity to save RAM."""
    global cached_model
    while True:
        try:
            cleanup_old_tasks() # Combine cleanups
            with cache_lock:
                if cached_model and (time.time() - last_activity_time > 300):
                    print("[*] Idle 5 mins: Purging Model Cache to free RAM.")
                    cached_model = None
                    gc.collect()
        except Exception as e:
            print(f"Watchdog error: {e}")
        time.sleep(60)

threading.Thread(target=cache_watchdog, daemon=True).start()

def get_model():
    """Smart Caching Model Loader."""
    global cached_model, last_activity_time
    with cache_lock:
        last_activity_time = time.time()
        if cached_model is None:
            if not os.path.exists(MODEL_XML):
                return None
            print("[*] Compiling Intel Engine (Initial Run)...")
            core = ov.Core()
            model = core.read_model(MODEL_XML)
            # LATENCY mode for instant single-image response
            cached_model = core.compile_model(model, "CPU", {"PERFORMANCE_HINT": "LATENCY"})
        return cached_model

def apply_premium_filters(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    gaussian_3 = cv2.GaussianBlur(img, (0, 0), 2.0)
    img = cv2.addWeighted(img, 1.5, gaussian_3, -0.5, 0)
    img = cv2.convertScaleAbs(img, alpha=1.05, beta=2)
    return img

def enhance_image(img, task_id):
    try:
        update_task(task_id, status='processing', message='Awakening AI Models...')
        model_infer = get_model()
        if model_infer is None:
            raise Exception("Model missing. Run converter.py.")

        start_time = time.time()
        max_dim = 512
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            sd = max_dim / max(h, w)
            img = cv2.resize(img, (int(w*sd), int(h*sd)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]

        tile_size, tile_pad, scale = 96, 10, 4
        img_HR = np.zeros((1, 3, h * scale, w * scale), dtype=np.uint8)
        
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
                elapsed = time.time() - start_time
                est_rem = int((total_tiles - tiles_done) * (elapsed / tiles_done))

                update_task(task_id, progress=max(2, progress), remaining=est_rem, 
                            message=f"Restoring Ultra-Sharp (Tile {tiles_done}/{total_tiles})")
                
                h_s, h_e = h_idx, min(h_idx + tile_size, h)
                w_s, w_e = w_idx, min(w_idx + tile_size, w)
                hs_p, he_p = max(h_s-tile_pad, 0), min(h_e+tile_pad, h)
                ws_p, we_p = max(w_s-tile_pad, 0), min(w_e+tile_pad, w)

                tile = img[hs_p:he_p, ws_p:we_p]
                tile = tile.astype(np.float32) / 255.0
                tile = np.transpose(tile[:, :, [2, 1, 0]], (2, 0, 1))
                tile = np.expand_dims(tile, axis=0)
                
                res = model_infer([tile])[output_layer]
                res_t = np.squeeze(res).clip(0, 1)
                res_t = (res_t * 255.0).round().astype(np.uint8)

                h_so = (h_s - hs_p) * scale
                h_eo = (h_e - hs_p) * scale
                w_so = (w_s - ws_p) * scale
                w_eo = (w_e - ws_p) * scale
                
                img_HR[0, :, h_s*scale:h_e*scale, w_s*scale:w_e*scale] = res_t[:, h_so:h_eo, w_so:w_eo]
                del res, res_t, tile
                gc.collect()

        del img
        gc.collect()

        output = np.squeeze(img_HR)
        del img_HR
        gc.collect()

        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        update_task(task_id, message='Perfecting Luminescence...')
        output = apply_premium_filters(output)
        
        result_path = os.path.join(RESULTS_DIR, f"{task_id}.jpg")
        cv2.imwrite(result_path, output, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        
        update_task(task_id, status='completed', progress=100, remaining=0, 
                    message='Masterpiece Ready', result_path=result_path)
        
    except Exception as e:
        print(f"Error {task_id}: {e}")
        update_task(task_id, status='error', message=f"Error: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'image' not in request.files: return jsonify({'error': 'No image'}), 400
        f = request.files['image']
        img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        task_id = str(uuid.uuid4())
        conn = sqlite3.connect(DB_PATH, timeout=20)
        c = conn.cursor()
        c.execute('PRAGMA journal_mode=WAL')
        c.execute("INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?)", 
                  (task_id, 'starting', 0, 0, 'Initializing Engine...', None, time.time()))
        conn.commit(); conn.close()
        threading.Thread(target=enhance_image, args=(img, task_id)).start()
        return jsonify({'task_id': task_id})
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/status/<task_id>')
def status(task_id):
    if task_id == 'undefined':
        return jsonify({'error': 'Invalid task ID'}), 400
    data = get_task(task_id)
    if not data: return jsonify({'error': 'Not found'}), 404
    resp = data.copy()
    if 'result_path' in resp: del resp['result_path']
    return jsonify(resp)

@app.route('/download/<task_id>')
def download(task_id):
    if task_id == 'undefined': return "Invalid ID", 400
    data = get_task(task_id)
    if not data or not data['result_path']: return "Not ready", 404
    return send_file(data['result_path'], mimetype='image/jpeg', as_attachment=True, download_name='clear.jpg')

@app.route('/image/<task_id>')
def get_image(task_id):
    if task_id == 'undefined': return "Invalid ID", 400
    data = get_task(task_id)
    if not data or not data['result_path']: return "Not ready", 404
    return send_file(data['result_path'], mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False, threaded=True, use_reloader=False)