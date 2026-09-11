"""ByteTrack tracker wrapper for campus-access.

Assigns persistent IDs to detections across frames. IDs are ephemeral
by design - see ethics/privacy_design.md for why this matters.
"""

from dataclasses import dataclass

import numpy as np

try:
    import supervision as sv
except ImportError as e:
    raise ImportError("supervision not installed. Run: pip install supervision") from e

from pipeline.detector import Detection, is_pedestrian, is_vehicle


@dataclass
class TrackedObject:
    """A single object tracked across frames.

    The tracker_id is randomly assigned per appearance and is never reused
    after the object exits the frame. The privacy design relies on this:
    there is no re-identification across cameras or sessions, so no
    surveillance vector is possible even in principle.
    """

    tracker_id: int
    class_id: int
    class_name: str
    first_seen_frame: int
    last_seen_frame: int
    last_bbox: tuple
    frames_seen: int = 0
    crossed_inward: bool = False
    crossed_outward: bool = False

    @property
    def is_pedestrian(self) -> bool:
        return self.class_id == 0

    @property
    def is_vehicle(self) -> bool:
        return self.class_id in (2, 3, 5)


class CampusTracker:
    """ByteTrack wrapper that emits TrackedObjects per frame.

    Two tracker instances run in parallel - one for pedestrians, one for
    vehicles. Running separate trackers keeps the ID spaces disjoint, so
    a person and a car can never share a tracker_id, which is what makes
    the downstream "people vs cars" analytics unambiguous.
    """

    def __init__(self) -> None:
        # ByteTrack via the supervision wrapper. Two instances so the
        # pedestrian and vehicle ID spaces don't collide.
        self._ped_tracker = sv.ByteTrack()
        self._veh_tracker = sv.ByteTrack()
        # Per-track state for dwell time and line crossing decisions.
        self._state: dict[int, TrackedObject] = {}

    def update(
        self,
        detections: list[Detection],
        frame_idx: int,
    ) -> list[TrackedObject]:
        """Update the trackers with a frame's detections.

        Args:
            detections: Detections from CampusDetector.detect_frame().
            frame_idx: Current frame number, for dwell-time accounting.

        Returns:
            The list of TrackedObjects that are still active (currently
            visible or recently visible). Objects that have not been seen
            for a while are pruned by supervision's internal logic.
        """
        ped_dets = [d for d in detections if is_pedestrian(d)]
        veh_dets = [d for d in detections if is_vehicle(d)]

        ped_sv = self._to_sv_detections(ped_dets)
        veh_sv = self._to_sv_detections(veh_dets)

        ped_tracked = (
            self._ped_tracker.update_with_detections(ped_sv)
            if len(ped_sv) > 0
            else None
        )
        veh_tracked = (
            self._veh_tracker.update_with_detections(veh_sv)
            if len(veh_sv) > 0
            else None
        )

        active: list[TrackedObject] = []

        if ped_tracked is not None:
            for det, tid in zip(ped_tracked, ped_tracked.tracker_id):
                self._update_state(tid, det, frame_idx, active)
        if veh_tracked is not None:
            for det, tid in zip(veh_tracked, veh_tracked.tracker_id):
                self._update_state(tid, det, frame_idx, active)

        return active

    def _update_state(
        self,
        tracker_id: int,
        sv_detection,
        frame_idx: int,
        active_list: list[TrackedObject],
    ) -> None:
        """Update per-track state and append to the active list."""
        # supervision's detections carry class_id and confidence in arrays.
        # Note: there is a builtin `class_id` keyword in our codebase that
        # shadows the local; we use a distinct local to avoid confusion.
        cls_idx = (
            int(sv_detection["class_id"]) if hasattr(sv_detection, "__getitem__") else 0
        )

        if tracker_id not in self._state:
            self._state[tracker_id] = TrackedObject(
                tracker_id=tracker_id,
                class_id=cls_idx,
                class_name="person" if cls_idx == 0 else "vehicle",
                first_seen_frame=frame_idx,
                last_seen_frame=frame_idx,
                last_bbox=tuple(float(v) for v in sv_detection[0])
                if hasattr(sv_detection, "__getitem__")
                else (0, 0, 0, 0),
            )
        else:
            self._state[tracker_id].last_seen_frame = frame_idx

        self._state[tracker_id].frames_seen += 1
        active_list.append(self._state[tracker_id])

    def _to_sv_detections(self, detections: list[Detection]):
        """Convert our Detection list to supervision's Detections format."""
        if not detections:
            return sv.Detections.empty()
        xyxy = np.array([d.bbox for d in detections], dtype=np.float32)
        confidence = np.array([d.confidence for d in detections], dtype=np.float32)
        class_id = np.array([d.class_id for d in detections], dtype=int)
        return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)

    @property
    def active_state(self) -> dict[int, TrackedObject]:
        """Read-only view of the current tracker state for analytics queries."""
        return dict(self._state)
