"""
GroundingDINO object detection with Gemini verification.
1. GroundingDINO detects candidate objects by text prompt
2. Gemini API verifies which detection is the correct target
"""

import warnings
warnings.filterwarnings("ignore", message=".*labels.*integer ids.*")

import base64
import json
import os
import re

import cv2
import numpy as np
import torch
from PIL import Image
from config import YOLO_CONFIDENCE, YOLO_DEVICE, C


_model = None
_processor = None
_gemini_client = None
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")

GDINO_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "grounding-dino-tiny")

def get_model():
    """Load GroundingDINO model (cached). Uses local path if available, else HuggingFace."""
    global _model, _processor
    if _model is None:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        model_id = GDINO_MODEL_PATH if os.path.isdir(GDINO_MODEL_PATH) else "IDEA-Research/grounding-dino-tiny"
        print(f"{C.BLUE}[detect]{C.RESET} Loading GroundingDINO ({model_id})...")
        _processor = AutoProcessor.from_pretrained(model_id)
        _model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        if YOLO_DEVICE == "cuda" and torch.cuda.is_available():
            _model = _model.to("cuda")
            print(f"{C.GREEN}[detect]{C.RESET} GroundingDINO loaded on CUDA")
        else:
            print(f"{C.GREEN}[detect]{C.RESET} GroundingDINO loaded on CPU")
    return _model, _processor


def get_gemini_client():
    """Get or create Gemini API client."""
    global _gemini_client
    if _gemini_client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None
        from google import genai
        _gemini_client = genai.Client(api_key=api_key)
        print(f"{C.GREEN}[detect]{C.RESET} Gemini verifier ready ({GEMINI_MODEL})")
    return _gemini_client


def annotate_numbered(frame, detections):
    """Draw numbered bounding boxes on frame for Gemini verification."""
    annotated = frame.copy()
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cx, cy = int(det["center"][0]), int(det["center"][1])
        color = (0, 255, 255)  # yellow for all candidates
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        # Large number label
        num = str(i + 1)
        cv2.putText(annotated, num, (x1 + 4, y1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(annotated, num, (x1 + 4, y1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        # Confidence
        label = f"{det['label']} {det['confidence']:.0%}"
        cv2.putText(annotated, label, (x1, y2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return annotated


def verify_with_gemini(frame, detections, target_label):
    """Ask Gemini to pick the correct detection from numbered candidates.

    Args:
        frame: BGR numpy array (original, un-annotated)
        detections: list of detection dicts from detect_objects
        target_label: what the user asked for (e.g. "red lego brick")

    Returns: index of the correct detection, or None if rejected/unavailable.
    """
    client = get_gemini_client()
    if client is None:
        return None  # no API key, skip verification

    if len(detections) == 0:
        return None

    # Annotate frame with numbered boxes
    annotated = annotate_numbered(frame, detections)

    # Encode as JPEG
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_bytes = buf.tobytes()

    # Build prompt
    box_list = "\n".join(
        f"  Box {i+1}: {d['label']} ({d['confidence']:.0%})"
        for i, d in enumerate(detections)
    )
    prompt = (
        f"I'm looking for: \"{target_label}\"\n\n"
        f"The image shows numbered bounding boxes around detected objects:\n{box_list}\n\n"
        f"Which numbered box contains the {target_label}? "
        f"Reply with ONLY a JSON object: {{\"box\": N}} where N is the box number (1-indexed), "
        f"or {{\"box\": 0}} if none of the boxes match."
    )

    try:
        from google.genai import types
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[types.Content(role="user", parts=[
                types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                types.Part.from_text(text=prompt),
            ])],
            config=types.GenerateContentConfig(
                system_instruction="You are a visual verification assistant for a robot arm. "
                    "Look at the numbered bounding boxes in the image and identify which one "
                    "contains the requested object. Reply with ONLY a JSON object, nothing else."
            )
        )
        match = re.search(r'\{.*?\}', response.text, re.DOTALL)
        if match:
            result = json.loads(match.group())
            box_num = int(result.get("box", 0))
            if 1 <= box_num <= len(detections):
                print(f"{C.GREEN}[gemini]{C.RESET} Verified: box {box_num} = {detections[box_num-1]['label']}")
                return box_num - 1  # convert to 0-indexed
            elif box_num == 0:
                print(f"{C.YELLOW}[gemini]{C.RESET} Rejected all detections — none match '{target_label}'")
                return None
        print(f"{C.YELLOW}[gemini]{C.RESET} Could not parse response: {response.text[:100]}")
        return None
    except Exception as e:
        print(f"{C.YELLOW}[gemini]{C.RESET} API error: {e}")
        return None


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


def detect_and_verify(frame, target_label):
    """Detect objects then verify with Gemini which one matches the target.

    Returns: (detections_list, verified_index) where verified_index is the
    index of the correct detection, or 0 if Gemini is unavailable/skipped.
    """
    detections = detect_objects(frame, target_label)
    if not detections:
        return [], None

    if len(detections) == 1:
        # Only one detection, no need to verify
        return detections, 0

    # Multiple detections — ask Gemini to pick
    verified_idx = verify_with_gemini(frame, detections, target_label)
    if verified_idx is not None:
        return detections, verified_idx

    # Gemini rejected all detections — return None to abort
    print(f"{C.YELLOW}[detect]{C.RESET} Gemini rejected all detections — aborting")
    return detections, None


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
