"""Line-zone crossing logic for campus-access.

A line zone is a virtual line drawn across the frame. When an object's
trajectory crosses the line, the system records the direction (inward
or outward). For campus access, "inward" = entering the building,
"outward" = exiting.

The line zone coordinates are in pixel space relative to the frame,
which means the user has to know their camera angle. The two-line pattern
(pedestrian gate line + vehicle gate line) is the common case.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

try:
    import supervision as sv
except ImportError as e:
    raise ImportError("supervision not installed. Run: pip install supervision") from e

from pipeline.tracker import TrackedObject


class CrossingDirection(Enum):
    """Which way an object crossed the line."""

    INWARD = "inward"  # Crossing toward the inside of the building
    OUTWARD = "outward"  # Crossing toward the outside


@dataclass
class CrossingEvent:
    """A single recorded crossing."""

    tracker_id: int
    class_name: str
    direction: CrossingDirection
    frame_idx: int
    is_pedestrian: bool
    is_vehicle: bool


class LineZone:
    """A virtual line that records inward/outward crossings.

    The line is defined by two endpoints (x1, y1) and (x2, y2) in pixel
    coordinates. The direction "inward" is whichever side of the line
    the user defines as inside the building.

    For a camera looking at a building entrance from the outside:
    - The line runs across the doorway horizontally.
    - Crossing top-to-bottom (or whichever direction "into the building"
      is) is recorded as INWARD.
    - Crossing the other way is OUTWARD.

    The implementation uses supervision's LineZone under the hood, which
    tracks per-tracker crossing state internally; we wrap it to emit
    CrossingEvent objects that carry the class metadata.
    """

    def __init__(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        name: str = "main",
    ) -> None:
        """Define a line zone.

        Args:
            start: (x, y) coordinates of one endpoint, in pixels.
            end: (x, y) coordinates of the other endpoint, in pixels.
            name: A label for the line zone (e.g. "pedestrian_gate",
                "vehicle_gate"). Used in CrossingEvents and reports.
        """
        self.name = name
        self.sv_zone = sv.LineZone(
            start=sv.Point(*start),
            end=sv.Point(*end),
            triggering_anchors=(sv.Position.CENTER,),
        )
        self._events: list[CrossingEvent] = []

    def update(
        self, tracked: list[TrackedObject], frame_idx: int
    ) -> list[CrossingEvent]:
        """Update the line zone with the current frame's tracked objects.

        supervision's LineZone.update() returns (in_count, out_count)
        as cumulative counters; we cross-reference with our own tracked
        state to emit CrossingEvents with class metadata.

        Args:
            tracked: Active TrackedObjects for this frame.
            frame_idx: Current frame number.

        Returns:
            A list of CrossingEvents recorded this frame (usually 0 or 1).
        """
        if not tracked:
            return []

        # Build a supervision Detections object from our tracked state.
        xyxy = np.array([t.last_bbox for t in tracked], dtype=np.float32)
        tracker_ids = np.array([t.tracker_id for t in tracked], dtype=int)
        sv_dets = sv.Detections(xyxy=xyxy, tracker_id=tracker_ids)

        # supervision's trigger returns the cumulative in/out counts.
        # We diff against the previous counts to detect this frame's events.
        in_before = self.sv_zone.in_count
        out_before = self.sv_zone.out_count

        self.sv_zone.trigger(sv_dets)

        new_events: list[CrossingEvent] = []
        if self.sv_zone.in_count > in_before:
            # Find which tracker crossed inward this frame.
            for t in tracked:
                if not t.crossed_inward:
                    t.crossed_inward = True
                    new_events.append(
                        CrossingEvent(
                            tracker_id=t.tracker_id,
                            class_name=t.class_name,
                            direction=CrossingDirection.INWARD,
                            frame_idx=frame_idx,
                            is_pedestrian=t.is_pedestrian,
                            is_vehicle=t.is_vehicle,
                        )
                    )
                    break  # one event per frame
        elif self.sv_zone.out_count > out_before:
            for t in tracked:
                if not t.crossed_outward:
                    t.crossed_outward = True
                    new_events.append(
                        CrossingEvent(
                            tracker_id=t.tracker_id,
                            class_name=t.class_name,
                            direction=CrossingDirection.OUTWARD,
                            frame_idx=frame_idx,
                            is_pedestrian=t.is_pedestrian,
                            is_vehicle=t.is_vehicle,
                        )
                    )
                    break

        self._events.extend(new_events)
        return new_events

    @property
    def total_in(self) -> int:
        return self.sv_zone.in_count

    @property
    def total_out(self) -> int:
        return self.sv_zone.out_count

    @property
    def events(self) -> list[CrossingEvent]:
        return list(self._events)
