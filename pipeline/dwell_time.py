"""Dwell-time tracking for campus-access.

Dwell time = how long an object has been visible since it entered.
For a person at a lecture hall, that's "how long have they been in class".
For a car at a parking gate, that's "how long has it been parked".

The tracker already records first_seen_frame / last_seen_frame per
tracked object. This module converts those frame indices into seconds,
given a known video frame rate, and exposes the dwell distribution as
an analytics-ready summary.
"""

from dataclasses import dataclass

from pipeline.tracker import TrackedObject


@dataclass
class DwellRecord:
    """A single object's completed dwell-time record."""

    tracker_id: int
    class_name: str
    is_pedestrian: bool
    is_vehicle: bool
    duration_seconds: float
    first_seen_frame: int
    last_seen_frame: int
    # Whether the object has been observed crossing a line zone.
    crossed_inward: bool
    crossed_outward: bool


@dataclass
class DwellSummary:
    """Aggregated dwell-time statistics for a session."""

    n_records: int
    n_pedestrians: int
    n_vehicles: int
    mean_dwell_seconds: float
    median_dwell_seconds: float
    min_dwell_seconds: float
    max_dwell_seconds: float
    still_present: int  # how many objects have not crossed outward yet


class DwellTimeAnalyzer:
    """Compute dwell time from the tracker's per-object state.

    The tracker emits TrackedObjects every frame; this analyzer watches
    the stream and, when an object is no longer visible (or has crossed
    outward), closes out its dwell record.
    """

    def __init__(self, fps: float) -> None:
        """Configure the dwell-time analyzer.

        Args:
            fps: Video frame rate. Dwell durations are reported in real
                seconds, so the fps value must match the source video.
        """
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        self.fps = fps
        # Per-tracker record state.
        self._open: dict[int, DwellRecord] = {}
        # Records that have been closed (object exited or crossed outward).
        self._closed: list[DwellRecord] = []

    def update(
        self,
        tracked: list[TrackedObject],
        frame_idx: int,
    ) -> None:
        """Update the analyzer with the current frame's tracked objects.

        Args:
            tracked: Active TrackedObjects for this frame.
            frame_idx: Current frame number.
        """
        active_ids = {t.tracker_id for t in tracked}

        # Close any open records whose object is no longer visible AND
        # has crossed a line (or simply disappeared).
        for tid in list(self._open.keys()):
            if tid not in active_ids:
                record = self._open.pop(tid)
                # The duration is recomputed at finalize time so we
                # don't need to recompute it here; we just close the
                # record. (The duration variable was a leftover from
                # an earlier draft that recomputed on close; keeping
                # the recompute in finalize() avoids drift if the
                # tracker later updates the record after close.)
                self._closed.append(record)

        # Open or update records for currently visible objects.
        for t in tracked:
            if t.tracker_id not in self._open:
                self._open[t.tracker_id] = DwellRecord(
                    tracker_id=t.tracker_id,
                    class_name=t.class_name,
                    is_pedestrian=t.is_pedestrian,
                    is_vehicle=t.is_vehicle,
                    duration_seconds=0.0,  # updated at close
                    first_seen_frame=t.first_seen_frame,
                    last_seen_frame=t.last_seen_frame,
                    crossed_inward=t.crossed_inward,
                    crossed_outward=t.crossed_outward,
                )
            else:
                # Update last-seen; duration is recomputed at close time.
                self._open[t.tracker_id].last_seen_frame = t.last_seen_frame
                self._open[t.tracker_id].crossed_inward = t.crossed_inward
                self._open[t.tracker_id].crossed_outward = t.crossed_outward

    def finalize(self) -> DwellSummary:
        """Close any remaining open records and produce the session summary.

        Returns:
            A DwellSummary aggregating all closed records plus any objects
            still visible at session end (counted as "still present").
        """
        # Close all remaining open records at their last-seen frame.
        still_present_records: list[DwellRecord] = []
        for record in self._open.values():
            record.duration_seconds = (
                record.last_seen_frame - record.first_seen_frame
            ) / self.fps
            still_present_records.append(record)
            self._closed.append(record)
        self._open.clear()

        if not self._closed:
            return DwellSummary(
                n_records=0,
                n_pedestrians=0,
                n_vehicles=0,
                mean_dwell_seconds=0.0,
                median_dwell_seconds=0.0,
                min_dwell_seconds=0.0,
                max_dwell_seconds=0.0,
                still_present=0,
            )

        durations = sorted(r.duration_seconds for r in self._closed)
        n = len(durations)
        median = (
            durations[n // 2]
            if n % 2 == 1
            else (durations[n // 2 - 1] + durations[n // 2]) / 2
        )

        return DwellSummary(
            n_records=len(self._closed),
            n_pedestrians=sum(1 for r in self._closed if r.is_pedestrian),
            n_vehicles=sum(1 for r in self._closed if r.is_vehicle),
            mean_dwell_seconds=sum(durations) / n,
            median_dwell_seconds=median,
            min_dwell_seconds=min(durations),
            max_dwell_seconds=max(durations),
            still_present=len(still_present_records),
        )

    @property
    def open_records(self) -> list[DwellRecord]:
        """Records for objects still visible in the most recent frame."""
        return list(self._open.values())

    @property
    def closed_records(self) -> list[DwellRecord]:
        return list(self._closed)
