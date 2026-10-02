# Real-Time Traffic Flow Analysis and Vehicle Counting System

An end-to-end computer vision application that detects, tracks, and counts mixed urban traffic flows from video streams and IP cameras using custom YOLO26s, ByteTrack, and geometric line-crossing algorithms.

---

## Demo & Application

### 1. Real-Time Demo

![Traffic System Demo](assets/demo.gif)

* Watch full demonstration video: [YouTube](https://youtu.be/rvv3OttnyZ4)

### 2. Standalone Windows Application
You can download the pre-packaged executable (9 GB package including PyTorch runtime and CUDA dependencies) without setting up a Python environment:

* Download: [TrafficAISystem.exe on Google Drive](https://drive.google.com/file/d/1Qr2T03t9HH3kCwu3H3Awu6Zh6lsrNOJ5/view?usp=sharing)
* Extract the `.zip` archive and run `TrafficAISystem.exe` directly.

---

## Key Features

* **Real-Time Detection:** Fine-tuned YOLO26s trained on 8,722 traffic images across 5 distinct classes: Car, Motorbike, Bus, Truck, and Bike.
* **Robust Multi-Object Tracking:** Integrated ByteTrack with Kalman filtering to maintain vehicle track IDs through occlusions and dense traffic.
* **Geometric Counting Logic:** Used bottom-center bounding box anchor points paired with Counter-Clockwise (CCW) line-crossing algorithms and polygon ROIs to eliminate double counting.
* **Desktop GUI:** Built with Tkinter and OpenCV, offering video file inspection, live camera stream processing, real-time bounding box visualization, and automated CSV logging.

---

## Quantitative Benchmarks

All inference benchmarks were measured on a workstation equipped with an NVIDIA GeForce RTX 2060 SUPER GPU (8 GB VRAM), CUDA 11.8, and PyTorch.

### 1. Detection Metrics (Test Set)

| Class | Precision | Recall | F1-Score | mAP@50 | Instances |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Car** | 0.9638 | 0.9332 | 0.9483 | 0.9731 | 5,202 |
| **Motorbike** | 0.9557 | 0.9338 | 0.9447 | 0.9647 | 6,221 |
| **Truck** | 0.9147 | 0.9063 | 0.9105 | 0.9487 | 1,020 |
| **Bus** | 0.9417 | 0.8383 | 0.8870 | 0.9299 | 207 |
| **Bike** | 0.8455 | 0.8507 | 0.8481 | 0.9124 | 784 |
| **Overall** | **0.9243** | **0.8925** | **0.9077** | **0.9458** | **13,434** |

### 2. Processing Latency

| Processing Stage | Latency (ms/frame) | Notes |
| :--- | :---: | :--- |
| Preprocessing | 2.1 ms | Image normalization |
| Tensor Inference | 8.5 ms | CUDA core execution |
| Postprocessing & Tracking | 1.0 ms | Decoding and ByteTrack association |
| **Total Pipeline** | **~12 ms** | **~83 FPS on RTX 2060 SUPER** |

---
Prerequisites

Python 3.10

NVIDIA GPU with CUDA 11.8 (optional, CPU supported with reduced FPS)

Installation:

Clone the repository

Install required packages

Launch the desktop GUI

---

## Repository Structure

```text
DoAnTGMT/
├── config/             # YAML configuration files for models and datasets
├── docs/               # Technical report (BaoCaoDoAnTGMT.pdf)
├── src/                # Core pipeline modules (detector, tracker, counter)
├── ui/                 # Tkinter GUI source scripts
├── weights/            # Model checkpoints (yolo26n.pt, yolo26s.pt)
├── main.py             # Main entry point for desktop GUI
├── train.py            # Training script with cosine annealing
└── requirements.txt    # Project dependencies
