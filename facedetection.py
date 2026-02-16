# src/face_detection.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as e:
    raise ImportError(
        "mediapipe is not installed. Run: pip install mediapipe"
    ) from e


@dataclass
class FaceDetectionResult:
    """Stores face bounding box in pixel coordinates and confidence score."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    score: float


class FaceDetector:
    """
    Real-time face detector using MediaPipe Face Detection.
    - Fast on CPU
    - Good for webcam pipelines
    """

    def __init__(
        self,
        model_selection: int = 0,
        min_detection_confidence: float = 0.6,
    ) -> None:
        """
        model_selection:
          0 = short-range (best for ~2m)
          1 = full-range (for farther faces)
        """
        self._mp_face = mp.solutions.face_detection
        self._detector = self._mp_face.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence,
        )

    def detect(
        self,
        frame_bgr: np.ndarray,
        max_faces: Optional[int] = 1,
    ) -> List[FaceDetectionResult]:
        """
        Detect faces in a BGR frame (OpenCV format).
        Returns a list of FaceDetectionResult sorted by confidence (desc).
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        h, w = frame_bgr.shape[:2]

        # MediaPipe expects RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._detector.process(frame_rgb)

        results: List[FaceDetectionResult] = []
        if not res.detections:
            return results

        for det in res.detections:
            score = float(det.score[0]) if det.score else 0.0
            bbox_rel = det.location_data.relative_bounding_box

            # Convert relative bbox to pixel bbox
            x1 = int(max(0, bbox_rel.xmin * w))
            y1 = int(max(0, bbox_rel.ymin * h))
            x2 = int(min(w, (bbox_rel.xmin + bbox_rel.width) * w))
            y2 = int(min(h, (bbox_rel.ymin + bbox_rel.height) * h))

            # Avoid invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue

            results.append(FaceDetectionResult(bbox=(x1, y1, x2, y2), score=score))

        # Sort by confidence and keep top faces if requested
        results.sort(key=lambda r: r.score, reverse=True)
        if max_faces is not None:
            results = results[:max_faces]
        return results

    @staticmethod
    def crop_face(
        frame_bgr: np.ndarray,
        bbox: Tuple[int, int, int, int],
        margin: float = 0.15,
        target_size: Optional[Tuple[int, int]] = (224, 224),
    ) -> np.ndarray:
        """
        Crop face from frame with optional margin and resize to target_size.
        margin is fraction of bbox size added on each side.
        """
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bbox

        bw = x2 - x1
        bh = y2 - y1
        pad_x = int(bw * margin)
        pad_y = int(bh * margin)

        xx1 = max(0, x1 - pad_x)
        yy1 = max(0, y1 - pad_y)
        xx2 = min(w, x2 + pad_x)
        yy2 = min(h, y2 + pad_y)

        face = frame_bgr[yy1:yy2, xx1:xx2].copy()
        if target_size is not None and face.size > 0:
            face = cv2.resize(face, target_size, interpolation=cv2.INTER_AREA)
        return face

    @staticmethod
    def draw_boxes(
        frame_bgr: np.ndarray,
        detections: List[FaceDetectionResult],
        show_score: bool = True,
    ) -> np.ndarray:
        """Draw bounding boxes and confidence on a copy of the frame."""
        out = frame_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if show_score:
                cv2.putText(
                    out,
                    f"{det.score:.2f}",
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
        return out


def demo_webcam() -> None:
    """
    Quick test: opens webcam, draws face bbox, shows cropped face.
    Press 'q' to quit.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Try changing VideoCapture index (0/1).")

    detector = FaceDetector(model_selection=0, min_detection_confidence=0.6)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections = detector.detect(frame, max_faces=1)

        vis = detector.draw_boxes(frame, detections)

        # Show cropped face in another window
        if detections:
            face = detector.crop_face(frame, detections[0].bbox, margin=0.15, target_size=(224, 224))
            cv2.imshow("Face Crop", face)

        cv2.imshow("Webcam - Face Detection", vis)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    demo_webcam()