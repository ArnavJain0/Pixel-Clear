# ✨ Pixel Clear: Ultra-Insight Engine

Pixel Clear is a professional-grade image restoration dashboard optimized specifically for **Intel CPU** performance via **OpenVINO**.

## 🚀 Why Pixel Clear?

Standard upscalers take forever on CPUs. Pixel Clear uses Intel's OpenVINO toolkit to achieve **2-5x faster inference** by compiling neural networks directly for your processor.

### Core Features:
- **Ultra-Insight Engine**: Beyond simple upscaling, we apply selective texture enhancement and intelligent lighting restoration.
- **Intel Optimization**: Native execution on Intel hardware—no NVIDIA GPU required.
- **Comparison Visualizer**: Professional side-by-side comparison slider.
- **Human-Centric UI**: A sleek, dark-mode dashboard with live ETA and phase tracking.

## 🛠️ Performance Architecture

Pixel Clear splits images into **512px tiles** with **10px padding** to ensure high-resolution images are processed without memory overflow. Our post-processing pipeline includes:
1. **Selective Contrast (CLAHE)**: Enhances local details without blowing out highlights.
2. **Interactive Sharpening**: Digitally reconstructs edges lost during blur.
3. **OpenVINO Inference**: Runs the SRGAN model at peak hardware efficiency.

## 💻 Running the Project

1. **Setup Environment**:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Convert Model (One-time Optimization)**:
   ```bash
   python converter.py
   ```

3. **Start Dashboard**:
   ```bash
   python app.py
   ```

## 🌐 Cloud Deployment (Render.com)

1. Connect your GitHub repo to Render.
2. Choose **Web Service**.
3. Use the following build command:
   ```bash
   pip install -r requirements.txt && python converter.py
   ```
4. Start command:
   ```bash
   gunicorn app:app
   ```
