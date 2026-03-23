# ─────────────────────────────────────────────────────────────
# Video / Frame Utilities
# Helpers for face detection and frame pre-processing.
# ─────────────────────────────────────────────────────────────
import cv2
import numpy as np

# Haar cascade for face detection (bundled with OpenCV)
_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_faces(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Detect face bounding boxes in a BGR frame.

    Returns:
        List of (x, y, w, h) tuples, one per detected face.
    """
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces  = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return [tuple(f) for f in faces] if len(faces) > 0 else []


def crop_face(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a face region from a BGR frame given a bounding box."""
    x, y, w, h = bbox
    return frame[y : y + h, x : x + w]


def largest_face(frame: np.ndarray) -> np.ndarray | None:
    """
    Detect faces and return the largest one as a BGR crop.
    Returns None if no face detected.
    """
    faces = detect_faces(frame)
    if not faces:
        return None
    bbox = max(faces, key=lambda b: b[2] * b[3])
    return crop_face(frame, bbox)


def draw_faces(frame: np.ndarray, emotion: str = "") -> np.ndarray:
    """Draw face bounding boxes and optional emotion label on a frame."""
    faces  = detect_faces(frame)
    output = frame.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 200, 100), 2)
        if emotion:
            cv2.putText(
                output, emotion.upper(),
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 100), 2,
            )
    return output
