"""
YOLO object detection wrapper.
Detects objects in camera frames and returns 2D pixel coordinates.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from config import YOLO_MODEL, YOLO_CONFIDENCE, YOLO_DEVICE, C


_model = None

def get_model():
    """Load YOLO model (cached)."""
    global _model
    if _model is None:
        print(f"{C.BLUE}[yolo]{C.RESET} Loading model: {YOLO_MODEL}...")
        _model = YOLO(YOLO_MODEL)
        # Warm up on GPU
        _model.predict(np.zeros((640, 640, 3), dtype=np.uint8), device=YOLO_DEVICE, verbose=False)
        print(f"{C.GREEN}[yolo]{C.RESET} Model loaded on {YOLO_DEVICE}")
    return _model


def detect_objects(frame, target_label=None):
    """Run YOLO detection on a frame.

    Args:
        frame: BGR numpy array (from OpenCV)
        target_label: optional string to filter by class name (e.g., "brick", "cup")

    Returns: list of detections, each a dict:
        {"label": str, "confidence": float, "bbox": (x1,y1,x2,y2), "center": (cx,cy)}
        Sorted by confidence (highest first).
    """
    model = get_model()
    results = model.predict(frame, device=YOLO_DEVICE, conf=YOLO_CONFIDENCE, verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2),
                "center": (cx, cy),
            })

    # Sort by confidence
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    # Filter by target label if specified
    if target_label:
        target_lower = target_label.lower()
        filtered = [d for d in detections if target_lower in d["label"].lower()]
        if filtered:
            return filtered
        # If no exact match, return all (user might describe it differently)

    return detections


def find_object_pixel(frame, target_label=None):
    """Detect the most likely target object and return its center pixel coordinate.

    Args:
        frame: BGR numpy array
        target_label: optional string to filter detections

    Returns: (cx, cy) pixel coordinates or None if nothing detected.
    """
    detections = detect_objects(frame, target_label)
    if not detections:
        return None
    best = detections[0]
    return best["center"]


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
    # Quick test with webcam
    import sys
    cam_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print(f"Opening camera {cam_idx}...")
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print("Failed to open camera")
        sys.exit(1)

    print("Running YOLO detection (press 'q' to quit)...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = detect_objects(frame)
        annotated = annotate_frame(frame, detections)
        for d in detections[:3]:
            print(f"  {d['label']:15s} {d['confidence']:.0%}  center=({d['center'][0]:.0f}, {d['center'][1]:.0f})")
        cv2.imshow("YOLO Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
