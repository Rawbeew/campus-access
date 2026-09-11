"""YOLO detector wrapper for campus-access.

Loads a YOLO model once and runs inference on a sequence of frames.
Returns detections in a normalized form for downstream tracking.
"""

from dataclasses import dataclass

import numpy as np

try:
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError("ultralytics not installed. Run: pip install ultralytics") from e


@dataclass
class Detection:
    """A single detected object in a frame."""

    class_id: int  # COCO class id (0=person, 1=bicycle, 2=car, ...)
    class_name: str  # Human-readable class name
    confidence: float  # Detection confidence in [0, 1]
    bbox: tuple  # (x_min, y_min, x_max, y_max) in pixels
    frame_idx: int  # Which frame this detection belongs to


# COCO class names we care about for campus access tracking.
# Mapping kept narrow on purpose: only the classes the system actually
# reports on. Adding more is a config change, not a code change.
CAMPUS_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
}


class CampusDetector:
    """YOLO wrapper that emits only campus-relevant classes.

    The detector itself is the standard ultralytics YOLO model; the wrapper
    exists so the rest of the pipeline never has to think about COCO ids,
    confidence thresholds, or device selection.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.4,
        device: str | None = None,
    ) -> None:
        """Load YOLO and configure the campus class filter.

        Args:
            model_path: Path to a YOLO model file. Defaults to yolov8n (nano,
                fastest on CPU). For accuracy, use yolov8s or yolov8m.
            confidence_threshold: Drop detections below this score.
                0.4 is a good default for crowded campus footage.
            device: Inference device. None lets ultralytics auto-select
                (CUDA if available, else CPU). Pass "cpu" to force CPU.
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = device

        # Restrict to campus-relevant classes at inference time.
        # YOLO accepts a `classes=[...]` argument that filters before
        # NMS, which is faster than filtering the output after the fact.
        self.class_ids = list(CAMPUS_CLASSES.keys())

    def detect_frame(self, frame: np.ndarray, frame_idx: int) -> list[Detection]:
        """Run detection on a single frame and return campus-relevant hits.

        Args:
            frame: BGR image (OpenCV convention).
            frame_idx: Frame number, attached to each Detection for downstream
                correlation with the tracker.

        Returns:
            A list of Detection objects. Empty if nothing was found.
        """
        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            classes=self.class_ids,
            device=self.device,
            verbose=False,
        )[0]

        detections: list[Detection] = []
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            detections.append(
                Detection(
                    class_id=cls_id,
                    class_name=CAMPUS_CLASSES.get(cls_id, f"class_{cls_id}"),
                    confidence=float(box.conf[0].item()),
                    bbox=tuple(float(v) for v in box.xyxy[0].tolist()),
                    frame_idx=frame_idx,
                )
            )
        return detections


def is_pedestrian(det: Detection) -> bool:
    """True if this detection is a person on foot (no vehicle)."""
    return det.class_id == 0


def is_vehicle(det: Detection) -> bool:
    """True if this detection is a motorized vehicle (car/bus/motorcycle).

    Bicycles are intentionally excluded: a person on a bicycle is, for
    campus access purposes, neither pedestrian nor vehicle. They show up
    in the raw detection stream but the analytics layer drops them so the
    "people vs cars" split is mutually exclusive.
    """
    return det.class_id in (2, 3, 5)
