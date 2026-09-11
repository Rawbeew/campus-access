"""Main entry point for campus-access.

Runs the full pipeline (detect -> track -> line-zone crossing -> dwell
analysis) on a video file and produces:
  1. An annotated output video (frames with detection boxes + line overlays)
  2. A session summary JSON (counts, dwell stats, peak info)
  3. A natural-language query interface for follow-up questions

Privacy is enforced via PrivacyConfig at startup. The system refuses
to run if on_device_only is True and any network call is attempted.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv2

from pipeline.detector import CampusDetector
from pipeline.dwell_time import DwellTimeAnalyzer
from pipeline.line_zones import LineZone
from pipeline.privacy import (
    NetworkBlockedError,
    PrivacyConfig,
    fail_if_cloud_call,
    load_default_privacy_config,
)
from pipeline.tracker import CampusTracker


def run_session(
    video_path: str,
    pedestrian_line: tuple,
    vehicle_line: tuple,
    output_video_path: str | None = None,
    output_summary_path: str | None = None,
    privacy_config: PrivacyConfig | None = None,
    yolo_model: str = "yolov8n.pt",
    confidence: float = 0.4,
) -> dict:
    """Run a campus-access session on one video file.

    Args:
        video_path: Path to the input video file.
        pedestrian_line: (start, end) coordinates for the pedestrian gate
            line zone, in pixels.
        vehicle_line: (start, end) coordinates for the vehicle gate line
            zone, in pixels.
        output_video_path: Where to write the annotated video. None skips
            the annotated output (faster).
        output_summary_path: Where to write the session summary JSON.
            None skips the summary output.
        privacy_config: Privacy configuration. None loads the strict
            defaults from load_default_privacy_config().
        yolo_model: YOLO model file to load. yolov8n (nano) is the
            default; yolov8s or yolov8m give better accuracy.
        confidence: Detection confidence threshold. Lower values catch
            more objects at the cost of more false positives.

    Returns:
        A dictionary with the session summary.

    Raises:
        NetworkBlockedError: If privacy_config.on_device_only is True and
            a network call is attempted.
    """
    if privacy_config is None:
        privacy_config = load_default_privacy_config()
    fail_if_cloud_call(privacy_config)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = CampusDetector(
        model_path=yolo_model,
        confidence_threshold=confidence,
    )
    tracker = CampusTracker()
    ped_zone = LineZone(
        start=pedestrian_line[0],
        end=pedestrian_line[1],
        name="pedestrian_gate",
    )
    veh_zone = LineZone(
        start=vehicle_line[0],
        end=vehicle_line[1],
        name="vehicle_gate",
    )
    dwell = DwellTimeAnalyzer(fps=fps)

    # Optional video writer for the annotated output.
    writer = None
    if output_video_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            output_video_path,
            fourcc,
            fps,
            (width, height),
        )

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect, track, and update line zones + dwell analyzer.
        detections = detector.detect_frame(frame, frame_idx)
        tracked = tracker.update(detections, frame_idx)

        # Pedestrian line zone gets only pedestrians; vehicle line gets only vehicles.
        ped_tracked = [t for t in tracked if t.is_pedestrian]
        veh_tracked = [t for t in tracked if t.is_vehicle]
        ped_zone.update(ped_tracked, frame_idx)
        veh_zone.update(veh_tracked, frame_idx)

        # Dwell is updated with all tracked (both pedestrian and vehicle).
        dwell.update(tracked, frame_idx)

        # Annotate and write the output frame.
        if writer is not None:
            annotated = frame.copy()
            for t in tracked:
                x1, y1, x2, y2 = (int(v) for v in t.last_bbox)
                color = (0, 255, 0) if t.is_pedestrian else (0, 0, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            # Draw the line zones.
            cv2.line(
                annotated,
                pedestrian_line[0],
                pedestrian_line[1],
                (0, 255, 0),
                2,
            )
            cv2.line(
                annotated,
                vehicle_line[0],
                vehicle_line[1],
                (0, 0, 255),
                2,
            )
            writer.write(annotated)

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    # Finalize and produce the summary.
    summary = dwell.finalize()
    result = {
        "video_path": video_path,
        "n_frames": frame_idx,
        "fps": fps,
        "resolution": {"width": width, "height": height},
        "pedestrian_zone": {
            "total_in": ped_zone.total_in,
            "total_out": ped_zone.total_out,
        },
        "vehicle_zone": {
            "total_in": veh_zone.total_in,
            "total_out": veh_zone.total_out,
        },
        "dwell_summary": asdict(summary),
        "privacy": {
            "on_device_only": privacy_config.on_device_only,
            "session_scoped_ids": privacy_config.session_scoped_ids,
            "no_per_person_persistence": privacy_config.no_per_person_persistence,
        },
    }

    if output_summary_path is not None:
        Path(output_summary_path).write_text(json.dumps(result, indent=2))

    return result


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="campus-access: privacy-preserving entry/exit counter for campus buildings"
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the input video file",
    )
    parser.add_argument(
        "--ped-line",
        required=True,
        help="Pedestrian gate line: x1,y1 x2,y2 (pixels)",
    )
    parser.add_argument(
        "--veh-line",
        required=True,
        help="Vehicle gate line: x1,y1 x2,y2 (pixels)",
    )
    parser.add_argument(
        "--output-video",
        default=None,
        help="Where to write the annotated output video (optional)",
    )
    parser.add_argument(
        "--output-summary",
        default=None,
        help="Where to write the session summary JSON (optional)",
    )
    parser.add_argument(
        "--yolo-model",
        default="yolov8n.pt",
        help="YOLO model file (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.4,
        help="Detection confidence threshold (default: 0.4)",
    )

    args = parser.parse_args()

    ped_start, ped_end = _parse_line_arg(args.ped_line)
    veh_start, veh_end = _parse_line_arg(args.veh_line)

    try:
        result = run_session(
            video_path=args.video,
            pedestrian_line=(ped_start, ped_end),
            vehicle_line=(veh_start, veh_end),
            output_video_path=args.output_video,
            output_summary_path=args.output_summary,
            yolo_model=args.yolo_model,
            confidence=args.confidence,
        )
        print(json.dumps(result, indent=2))
        return 0
    except NetworkBlockedError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


def _parse_line_arg(arg: str) -> tuple:
    """Parse 'x1,y1 x2,y2' into two (x, y) tuples."""
    parts = arg.split()
    if len(parts) != 2:
        raise ValueError(f"Line must be 'x1,y1 x2,y2', got: {arg}")
    a = tuple(int(v) for v in parts[0].split(","))
    b = tuple(int(v) for v in parts[1].split(","))
    if len(a) != 2 or len(b) != 2:
        raise ValueError(f"Each endpoint must be 'x,y', got: {arg}")
    return a, b


if __name__ == "__main__":
    sys.exit(main())
