# 🎯 Real-Time Object Detection with Audio Feedback using YOLOv3

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-YOLOv3-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-DNN-green.svg)

---

## 📌 Abstract

This project implements a **real-time object detection system** using **YOLOv3 (You Only Look Once)** to identify objects from a live webcam feed and provide **audio feedback** describing the detected objects and their spatial positions. The system is designed to demonstrate **real-time computer vision**, confidence-based filtering, and **accessibility-focused AI applications**.

---

## 🎯 Objectives

- **Real-Time Detection:** Detect multiple objects from live video streams with low latency  
- **Audio Feedback:** Convert detected object information into speech for user awareness  
- **Accessibility:** Assist visually impaired users by providing spoken descriptions of surroundings  
- **Efficiency:** Optimize detection using confidence thresholding and Non-Maximum Suppression  

---

## 🛠️ Methodology

### 1️⃣ Video Input & Frame Capture
- Captured live video frames using a webcam through **OpenCV**
- Processed frames at regular intervals to balance performance and accuracy

---

### 2️⃣ Image Preprocessing
- Converted frames into **blob format** using OpenCV’s DNN module
- Normalized pixel values and resized images to YOLO’s input size (416×416)

---

### 3️⃣ Object Detection using YOLOv3
- Used **YOLOv3**, a single-stage object detector, for real-time inference
- Predicted bounding boxes, class probabilities, and confidence scores
- Applied a confidence threshold to filter weak detections

---

### 4️⃣ Non-Maximum Suppression (NMS)
- Removed overlapping bounding boxes using **Non-Maximum Suppression**
- Retained only the most confident detections for each object

---

### 5️⃣ Spatial Position Analysis
- Divided the frame into logical regions:
  - Left / Center / Right  
  - Top / Middle / Bottom  
- Determined the relative position of each detected object

---

### 6️⃣ Audio Feedback Generation
- Converted detected object labels and positions into text
- Generated spoken output using **Text-to-Speech (gTTS)** for real-time feedback

---

## 📊 Results & Observations

- Successfully detected common objects such as people, chairs, bottles, and mobile phones
- Provided clear and timely audio feedback describing object positions
- Maintained real-time performance with optimized frame processing
- Demonstrated effective object localization using YOLOv3

---

## 🚀 How to Use

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Saravanan-S-315/Real-Time-Object-Detection-with-Audio-Feedback-YOLOv3.git
cd Real-Time-Object-Detection-with-Audio-Feedback-YOLOv3
