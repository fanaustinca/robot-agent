"""
GroundingDINO object detection wrapper.
Detects objects in camera frames using open-vocabulary text prompts.
"""

import warnings
warnings.filterwarnings("ignore", message=".*labels.*integer ids.*")

import cv2
import numpy as np
import torch
from PIL import Image
from config import YOLO_CONFIDENCE, YOLO_DEVICE, C


_model = None
_processor = None

def get_model():
    """Load GroundingDINO model (cached)."""
    global _model, _processor
    if _model is None:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        model_id = "IDEA-Research/grounding-dino-tiny"
        print(f"{C.BLUE}[detect]{C.RESET} Loading GroundingDINO ({model_id})...")
        _processor = AutoProcessor.from_pretrained(model_id)
        _model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        if YOLO_DEVICE == "cuda" and torch.cuda.is_available():
            _model = _model.to("cuda")
            print(f"{C.GREEN}[detect]{C.RESET} GroundingDINO loaded on CUDA")
        else:
            print(f"{C.GREEN}[detect]{C.RESET} GroundingDINO loaded on CPU")
    return _model, _processor


def detect_objects(frame, target_label=None):
    """Run GroundingDINO detection on a frame.

    Args:
        frame: BGR numpy array (from OpenCV)
        target_label: text description of what to detect (e.g., "red block", "brick")

    Returns: list of detections, each a dict:
        {"label": str, "confidence": float, "bbox": (x1,y1,x2,y2), "center": (cx,cy)}
        Sorted by confidence (highest first).
    """
    model, processor = get_model()
    device = next(model.parameters()).device

    # Convert BGR to RGB PIL image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)

    # Build text prompt — GroundingDINO uses "." separated phrases
    if target_label:
        text = target_label.strip().rstrip(".") + "."
    else:
        text = "object. block. cube. ball. cup. bottle. box."

    inputs = processor(images=pil_image, text=text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        input_ids=inputs.input_ids,
        threshold=YOLO_CONFIDENCE,
        text_threshold=YOLO_CONFIDENCE,
        target_sizes=[pil_image.size[::-1]],  # (height, width)
    )[0]

    detections = []
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        x1, y1, x2, y2 = box.tolist()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        detections.append({
            "label": label,
            "confidence": float(score),
            "bbox": (x1, y1, x2, y2),
            "center": (cx, cy),
        })

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def find_object_pixel(frame, target_label=None):
    """Detect the most likely target object and return its center pixel coordinate."""
    detections = detect_objects(frame, target_label)
    if not detections:
        return None
    return detections[0]["center"]


def annotate_frame(frame, detections, target_idx=0):
    """Draw detection boxes on a frame copy. Returns annotated frame."""
    annotated = frame.copy()
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cx, cy = int(det["center"][0]), int(det["center"][1])
        color = (0, 255, 0) if i == target_idx else (0, 165, 255)
        thickness = 2 if i == target_idx else 1

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        label = f"{det['label']} {det['confidence']:.0%}"
        cv2.putText(annotated, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.circle(annotated, (cx, cy), 5, color, -1)

    return annotated


if __name__ == "__main__":
    import sys
    cam_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    label = sys.argv[2] if len(sys.argv) > 2 else None
    print(f"Opening camera {cam_idx}...")
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print("Failed to open camera")
        sys.exit(1)

    print(f"Running GroundingDINO detection (label={label}, press 'q' to quit)...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = detect_objects(frame, label)
        annotated = annotate_frame(frame, detections)
        for d in detections[:3]:
            print(f"  {d['label']:15s} {d['confidence']:.0%}  center=({d['center'][0]:.0f}, {d['center'][1]:.0f})")
        cv2.imshow("GroundingDINO Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
