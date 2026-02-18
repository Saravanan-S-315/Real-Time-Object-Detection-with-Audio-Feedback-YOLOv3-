# 🎯 Real-Time Object Detection with Audio Feedback (YOLOv3)

This project runs **YOLOv3 object detection on a webcam feed** and announces detected objects with their approximate screen positions (e.g., `top left person`).

## Features

- Real-time YOLOv3 inference with OpenCV DNN
- Confidence filtering + Non-Maximum Suppression (NMS)
- Relative spatial localization (`left/center/right`, `top/middle/bottom`)
- Configurable speech cooldown to avoid repeated announcements
- CLI arguments for thresholds, model paths, and camera source

## Project structure

- `main.py` — main application entrypoint
- `yolov3.cfg`, `yolov3.weights`, `coco.names` — required YOLO model artifacts (download separately)

## Setup

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Place YOLO files in project root:

- `yolov3.cfg`
- `yolov3.weights`
- `coco.names`

## Run

```bash
python main.py --source 0 --confidence 0.5 --nms-threshold 0.4 --detect-every 10 --tts-cooldown 3
```

Press `q` to quit.

## Notes

- If `gTTS` or `playsound` are missing/unavailable, the app will continue running and print speech text instead of playing audio.
- `gTTS` uses Google Text-to-Speech and may require network access.
