# ✨ Pixel Clear: High-Performance AI Image Restoration

**Pixel Clear** is a professional-grade image restoration dashboard optimized for high-performance, low-RAM execution. It uses the **FSRCNN** neural network for 4x super-resolution and custom filters for high-quality deblurring.

---

## 🚀 Key Features

*   **PCA-Optimized Pipeline**: Memory-efficient architecture that isolates heavy AI tasks to short-lived subprocesses, ensuring it runs perfectly on free-tier hosting (like Render.com 512MB RAM).
*   **Edge Computing**: Client-side resizing via HTML5 Canvas prevents large file uploads and server crashes.
*   **Professional Comparison Visualizer**: Interactive side-by-side slider for real-time quality comparison.
*   **Modern Dashboard**: Sleek, dark-mode UI with live progress tracking and built-in interactive Snake game for wait times.

---

## 🛠️ Performance Architecture

1.  **AI Reconstruction (`FSRCNN x4`)**: Uses a lightweight neural network (88KB) to reconstruct missing details, providing high-quality upscaling with minimal resource cost.
2.  **Selective Contrast (CLAHE)**: Normalizes lighting and enhances local textures without overexposing highlights.
3.  **Unsharp Masking**: Mathematically sharpened edges to provide a crisp, deblurred look.
4.  **Subprocess Isolation**: The AI model is only loaded into the RAM during the seconds it's actively working, completely freeing memory on task completion.

---

## 💻 Running the Project Locally

1.  **Setup Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

2.  **Start Dashboard**:
    ```bash
    uvicorn server:app --reload --port 5000
    ```

3.  **Open Browser**: `http://localhost:5000`

---

## 🌐 Cloud Deployment (Render.com)

1.  Connect your GitHub repo to **Render**.
2.  Choose **Web Service**.
3.  **Build Command**: `pip install -r requirements.txt`  
    *(Note: Remove any reference to `python converter.py` as it is now redundant.)*
4.  **Start Command**: `uvicorn server:app --host 0.0.0.0 --port $PORT`

---

## 🎓 For Viva Preparation
See [about.md](about.md) for a complete technical guide, file-by-file breakdown, and expected questions/answers on the project's internal mechanics and AI implementation.
