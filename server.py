# ============================================================
# server.py  – Pixel-Clear web service
# ------------------------------------------------------------
# This is the ONLY process that runs permanently on Render.
# It imports NOTHING heavy (no cv2, no numpy, no openvino).
# All actual image processing is delegated to worker.py, which
# runs as a short-lived subprocess and exits immediately, so
# the OS reclaims 100% of the RAM it used.
# ============================================================
import os
import sys
import uuid
import time
import threading
import subprocess
import sqlite3

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ── Configuration ────────────────────────────────────────────
DB_PATH              = "tasks.db"
RESULTS_DIR          = "static/results"
TMP_DIR              = "/tmp"
MAX_UPLOAD_BYTES     = 5 * 1024 * 1024   # 5 MB – the browser already resized
INFER_SEMAPHORE      = threading.Semaphore(1)  # one heavy job at a time
# ─────────────────────────────────────────────────────────────

os.makedirs(RESULTS_DIR, exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ── SQLite helpers ────────────────────────────────────────────
def _conn():
    c = sqlite3.connect(DB_PATH, timeout=20)
    c.execute("PRAGMA journal_mode=WAL")
    return c

def init_db():
    db = _conn()
    db.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id          TEXT PRIMARY KEY,
            status      TEXT,
            progress    INTEGER,
            remaining   INTEGER,
            message     TEXT,
            result_path TEXT,
            timestamp   REAL
        )""")
    db.commit()
    db.close()

init_db()

def update_task(task_id: str, **fields):
    if not fields:
        return
    db = _conn()
    cols = ", ".join(f"{k}=?" for k in fields)
    db.execute(f"UPDATE tasks SET {cols} WHERE id=?",
               list(fields.values()) + [task_id])
    db.commit()
    db.close()

def get_task(task_id: str):
    db = _conn()
    cur = db.execute(
        "SELECT status, progress, remaining, message, result_path FROM tasks WHERE id=?",
        (task_id,))
    row = cur.fetchone()
    db.close()
    if row:
        return dict(zip(
            ["status", "progress", "remaining", "message", "result_path"], row))
    return None

def cleanup_old_tasks():
    cutoff = time.time() - 3600
    db = _conn()
    rows = db.execute(
        "SELECT result_path FROM tasks WHERE timestamp < ?", (cutoff,)).fetchall()
    for (path,) in rows:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
    db.execute("DELETE FROM tasks WHERE timestamp < ?", (cutoff,))
    db.commit()
    db.close()

def _watchdog():
    while True:
        try:
            cleanup_old_tasks()
        except Exception as exc:
            print(f"[watchdog] {exc}")
        time.sleep(300)   # run cleanup every 5 minutes

threading.Thread(target=_watchdog, daemon=True).start()

# ── Subprocess worker ─────────────────────────────────────────
def _run_worker(input_path: str, output_path: str, task_id: str):
    try:
        update_task(task_id, status="processing", message="In queue…", progress=5)
        with INFER_SEMAPHORE:
            update_task(task_id, message="Running AI model…", progress=15)
            result = subprocess.run(
                [sys.executable, "worker.py", input_path, output_path],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or "worker failed")
            update_task(task_id,
                        status="completed", progress=100, remaining=0,
                        message="Masterpiece Ready", result_path=output_path)
    except Exception as exc:
        print(f"[worker-thread] {exc}")
        update_task(task_id, status="error", message=str(exc))
    finally:
        # always clean up the temp file
        try:
            os.remove(input_path)
        except OSError:
            pass

# ── Routes ────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload(image: UploadFile = File(...)):
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(400, "Unsupported file type")
    data = await image.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (max 5 MB)")

    task_id    = str(uuid.uuid4())
    input_path = os.path.join(TMP_DIR,     f"{task_id}_in.jpg")
    out_path   = os.path.join(RESULTS_DIR, f"{task_id}.jpg")

    with open(input_path, "wb") as fh:
        fh.write(data)

    db = _conn()
    db.execute("INSERT INTO tasks VALUES (?,?,?,?,?,?,?)",
               (task_id, "starting", 0, 0, "Waking up…", None, time.time()))
    db.commit()
    db.close()

    threading.Thread(target=_run_worker,
                     args=(input_path, out_path, task_id),
                     daemon=True).start()

    return JSONResponse({"task_id": task_id})


@app.get("/status/{task_id}")
async def status(task_id: str):
    data = get_task(task_id)
    if not data:
        raise HTTPException(404, "Task not found")
    data.pop("result_path", None)
    return JSONResponse(data)


@app.get("/image/{task_id}")
async def get_image(task_id: str):
    data = get_task(task_id)
    if not data or not data.get("result_path"):
        raise HTTPException(404, "Image not ready")
    return FileResponse(data["result_path"], media_type="image/jpeg")


@app.get("/download/{task_id}")
async def download(task_id: str):
    data = get_task(task_id)
    if not data or not data.get("result_path"):
        raise HTTPException(404, "File not ready")
    return FileResponse(data["result_path"],
                        media_type="image/jpeg",
                        filename="clear.jpg")
