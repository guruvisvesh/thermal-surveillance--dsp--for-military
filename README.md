# 🔥 Thermal Surveillance System using DSP & Sensor Fusion

## 📌 Overview
This project implements a **thermal surveillance and object detection system** by combining **2D image processing** and **1D sensor signal processing** using Digital Signal Processing (DSP) techniques.

It integrates:
- Thermal image analysis (FLIR dataset)
- Real-time IR sensor data (MLX90614 via Arduino)
- Frequency-domain and time-domain processing

---

## 🎯 Key Features

### 🖼️ Thermal Image Processing (2D DSP)
- Grayscale conversion & normalization
- Histogram equalization for contrast enhancement
- 2D FFT for frequency analysis
- Edge detection using convolution filters
- Feature extraction (mean, max, std, hotspots)

---

### 🔊 Sensor Signal Processing (1D DSP)
- Real-time temperature acquisition (Arduino)
- FIR filtering for noise reduction
- Time-series signal analysis

---

### 🤖 Object Detection & Classification
- Contour-based detection
- Feature-based classification:
  - Human
  - Vehicle
- Optional ML model (Random Forest)

---

### 🔗 Sensor + Image Fusion
- Correlates image intensity with sensor temperature
- Improves detection confidence
- Reduces false positives

---

## 🧠 Concepts Used
- Digital Signal Processing (DSP)
- 2D Discrete Fourier Transform (DFT)
- FIR Filtering
- Convolution
- Feature Extraction
- Computer Vision

---

## ⚙️ Tech Stack
- Python (NumPy, OpenCV, SciPy, Pandas)
- Arduino (MLX90614 IR Sensor)
- Matplotlib
- Scikit-learn (optional ML)

---

## 📊 Results
- Accurate detection of humans & vehicles
- Temperature correlation within ±5–8°C
- Improved detection reliability in low-visibility conditions

---

## 🚀 How to Run

```bash
pip install numpy opencv-python scipy pandas matplotlib scikit-learn pyserial
python src/main_pipeline.py
