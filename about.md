# 💎 Pixel-Clear | The "PCA-Optimized" AI Documentation

**A complete technical guide for understanding the High-Performance, Low-RAM Image Restoration Engine.**

---

## 🏎️ 1. The ML "PCA" Analogy
In Machine Learning, **Principal Component Analysis (PCA)** is the art of reducing complexity while keeping the most important "signal." 

**Pixel-Clear follows this exact philosophy:**
1.  **Discarding the "Noise"**: We removed massive external dependencies (like `torch` and `openvino`) from the permanent server process. They are like "redundant dimensions" that take up 400MB+ of RAM for no reason.
2.  **Keeping the "Principal Component"**: We isolated the AI "heavy lifting" into a single "one-shot" subprocess. This script captures 100% of the value (the enhancement) but only exists in RAM for the few seconds it's actually working. 
3.  **Data Normalization**: Just as PCA requires normalized data, we normalize our images (resizing to ≤ 480px) in the user's browser *before* they reach the server. This prevents any "outlier" (a huge 50MB file) from crashing our memory-limited server.

---

## 🛠️ 2. Viva Essentials: The "Where & How" of Data

### 📂 Storage & Uploads
*   **Where is the image uploaded?** 
    It is uploaded to the server's local **`/tmp`** directory. We save the raw bytes from the UI into a temporary `.jpg` file using a unique `task_id` (so multiple users don't overwrite each other).
*   **Where is it stored?** 
    - **Originals**: Stay in `/tmp` and are deleted immediately after the AI finishes.
    - **Enhanced Results**: Are stored in the **`static/results/`** directory on the server.
*   **Is there a cleanup process?** 
    Yes. The **`_watchdog()`** function runs in the background. It checks the folder every 5 minutes and deletes any image that is older than **1 hour** to keep the server's storage free.

### ⬇️ Downloads
*   **How are you able to download?** 
    The UI calls a dedicated route (**`download`**). This function checks the **SQLite database (`tasks.db`)** to find the file path for that user's specific `task_id` and then sends that file back as a "downloadable" attachment.

---

## 🧠 3. The "Heavy Lifting": How Deblurring Works

When you see the image transform from blurry to sharp, these two functions are doing the work:

### 1. `run_ai_upscale()` (The Neural Brain)
This function uses the **FSRCNN x4 (Fast Super-Resolution CNN)** model. 
- **The Magic**: Instead of just blowing up the pixels (which makes them look pixelated), the neural network "hallucinates" or recreates the missing details. It has been trained on millions of images to know what sharp edges should look like, effectively "reconstructing" the high-res version.

### 2. `apply_filters()` (The Finishing Surgeon)
After the AI upscales the image, this function refines it to look professional:
- **`CLAHE` (Contrast Enhancement)**: It fixes the lighting and normalization across the image so the colors pop.
- **`Unsharp Masking` (Sharpening)**: It uses a mathematical "subtraction" method (Gaussian Blur subtraction) to highlight the edges. This is what mathematically "deblurs" the image by increasing the contrast at every edge.

---

## 🖼️ 4. UI Mechanics: What the User Sees vs. What Happens

| UI Feature | Function/Module Responsible | What is actually happening? |
| :--- | :--- | :--- |
| **Comparison Slider** | **CSS + JS (`initSlider`)** | Two images (`img-before` and `img-after`) are layered. A transparent "handle" clips the top image's width as you drag it. |
| **Progress Bar** | **`pollStatus()` Polling** | The browser asks the server every 1s: "What is the status of Task X?". The server reads the **SQLite database** and sends back a number (e.g., 85%). |
| **Fast Interaction** | **`resizeImage()` (JS)** | We use the **HTML5 Canvas** on the user's browser to shrink the image *before* upload. This offloads the heavy work from our server to the user's computer ("Edge Computing"). |

---

## 🚀 5. Deployment Troubleshooting
> [!IMPORTANT]
> **Build Error: "python: can't open file 'converter.py'"**
> If your deployment fails on Render with this error, it's because your **Build Command** in the Render Dashboard is still trying to run the old converter script.
> 
> **The Fix**: 
> Go to your **Render Dashboard** -> **Settings** -> **Build Command** and change it to simply:
> `pip install -r requirements.txt`
> 
> (The new architecture handles model setup automatically inside `worker.py`, so `converter.py` is no longer needed).

---

## 🌟 6. Summary for Viva
Pixel-Clear is a **Zero-Waste** application. Every byte of RAM is accounted for by using **subprocess isolation** (killing the worker after use) and **client-side normalization** (resizing before upload). This ensures that heavy AI models can run on a tiny 512MB server without ever crashing.
