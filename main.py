import argparse
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np

try:
    from gtts import gTTS
except ImportError:  # optional dependency at runtime
    gTTS = None

try:
    from playsound import playsound
except ImportError:  # optional dependency at runtime
    playsound = None


@dataclass
class Detection:
    label: str
    confidence: float
    box: Tuple[int, int, int, int]
    center: Tuple[int, int]


@dataclass
class Config:
    cfg: Path
    weights: Path
    labels: Path
    source: int
    confidence_threshold: float
    nms_threshold: float
    detect_every_n_frames: int
    tts_cooldown_seconds: float
    window_name: str


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Real-time YOLOv3 object detection with positional audio feedback"
    )
    parser.add_argument("--cfg", default="yolov3.cfg", type=Path)
    parser.add_argument("--weights", default="yolov3.weights", type=Path)
    parser.add_argument("--labels", default="coco.names", type=Path)
    parser.add_argument("--source", default=0, type=int, help="Camera source index")
    parser.add_argument("--confidence", default=0.5, type=float)
    parser.add_argument("--nms-threshold", default=0.4, type=float)
    parser.add_argument("--detect-every", default=10, type=int)
    parser.add_argument("--tts-cooldown", default=3.0, type=float)
    parser.add_argument("--window-name", default="YOLOv3 Object Detection + Audio")

    args = parser.parse_args()
    return Config(
        cfg=args.cfg,
        weights=args.weights,
        labels=args.labels,
        source=args.source,
        confidence_threshold=args.confidence,
        nms_threshold=args.nms_threshold,
        detect_every_n_frames=max(args.detect_every, 1),
        tts_cooldown_seconds=max(args.tts_cooldown, 0.0),
        window_name=args.window_name,
    )


def validate_files(config: Config) -> None:
    missing = [
        path for path in (config.cfg, config.weights, config.labels) if not path.exists()
    ]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Required model files are missing: {missing_text}. "
            "Download YOLOv3 cfg/weights and coco.names before running."
        )


def load_classes(labels_path: Path) -> List[str]:
    with labels_path.open("r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def get_output_layers(net: cv2.dnn.Net) -> List[str]:
    layer_names = net.getLayerNames()
    output_layers = net.getUnconnectedOutLayers()
    return [layer_names[index - 1] for index in output_layers.flatten()]


def detect_objects(
    frame: np.ndarray,
    net: cv2.dnn.Net,
    output_layers: Sequence[str],
    classes: Sequence[str],
    confidence_threshold: float,
    nms_threshold: float,
) -> List[Detection]:
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(
        frame,
        scalefactor=1 / 255.0,
        size=(416, 416),
        swapRB=True,
        crop=False,
    )

    net.setInput(blob)
    detections = net.forward(output_layers)

    boxes: List[List[int]] = []
    confidences: List[float] = []
    class_ids: List[int] = []
    centers: List[Tuple[int, int]] = []

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])

            if confidence < confidence_threshold:
                continue

            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            box_width = int(detection[2] * width)
            box_height = int(detection[3] * height)
            x = int(center_x - box_width / 2)
            y = int(center_y - box_height / 2)

            boxes.append([x, y, box_width, box_height])
            confidences.append(confidence)
            class_ids.append(class_id)
            centers.append((center_x, center_y))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    results: List[Detection] = []

    if len(indices) == 0:
        return results

    for index in indices.flatten():
        x, y, box_width, box_height = boxes[index]
        results.append(
            Detection(
                label=classes[class_ids[index]],
                confidence=confidences[index],
                box=(x, y, box_width, box_height),
                center=centers[index],
            )
        )

    return results


def get_position(center: Tuple[int, int], frame_width: int, frame_height: int) -> str:
    cx, cy = center
    horizontal = "left" if cx < frame_width / 3 else "center" if cx < 2 * frame_width / 3 else "right"
    vertical = "top" if cy < frame_height / 3 else "middle" if cy < 2 * frame_height / 3 else "bottom"
    return f"{vertical} {horizontal}"


def draw_detections(frame: np.ndarray, detections: Sequence[Detection], classes: Sequence[str]) -> List[str]:
    height, width, _ = frame.shape
    rng = np.random.default_rng(seed=42)
    colors = rng.uniform(0, 255, size=(len(classes), 3))

    descriptions: List[str] = []
    for item in detections:
        x, y, box_width, box_height = item.box
        class_index = classes.index(item.label)
        color = colors[class_index]

        cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), color, 2)
        cv2.putText(
            frame,
            f"{item.label} {item.confidence:.2f}",
            (x, max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        position = get_position(item.center, width, height)
        descriptions.append(f"{position} {item.label}")

    return descriptions


def speak(text: str) -> bool:
    if gTTS is None or playsound is None:
        print(f"[WARN] Text-to-speech libraries unavailable. Would say: {text}")
        return False

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        temp_path = temp_audio.name

    try:
        tts = gTTS(text=text, lang="en")
        tts.save(temp_path)
        playsound(temp_path)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to play TTS audio: {exc}")
        return False
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def run(config: Config) -> None:
    validate_files(config)
    classes = load_classes(config.labels)
    net = cv2.dnn.readNet(str(config.weights), str(config.cfg))
    output_layers = get_output_layers(net)

    cap = cv2.VideoCapture(config.source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {config.source}")

    print("[INFO] YOLOv3 model loaded successfully")
    frame_count = 0
    last_spoken_time = 0.0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("[WARN] Failed to read frame from camera")
                break

            frame_count += 1
            if frame_count % config.detect_every_n_frames == 0:
                detections = detect_objects(
                    frame,
                    net,
                    output_layers,
                    classes,
                    config.confidence_threshold,
                    config.nms_threshold,
                )
                spoken_descriptions = draw_detections(frame, detections, classes)

                now = time.time()
                if (
                    spoken_descriptions
                    and now - last_spoken_time >= config.tts_cooldown_seconds
                ):
                    speech_text = "Detected " + ", ".join(spoken_descriptions)
                    if speak(speech_text):
                        last_spoken_time = now

            cv2.imshow(config.window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    config = parse_args()
    run(config)


if __name__ == "__main__":
    main()
