"""Evaluation benchmark for campus-access.

Measures detection and tracking accuracy on a held-out test video.
Reports precision, recall, and the MOTA-style identity-switch count for
both pedestrian and vehicle tracks.

Usage:
    python eval/benchmark.py --video test_video.mp4 \\
        --ped-line "0,540 1920,540" \\
        --veh-line "0,540 1920,540" \\
        --ground-truth ground_truth.json

The ground-truth JSON has the schema:
    {
      "frames": [
        {"frame_idx": 0, "pedestrians": [{"id": "p1", "bbox": [x1, y1, x2, y2]}], "vehicles": []},
        ...
      ]
    }

For now the benchmark is a stub that measures throughput; ground-truth
comparison is left as a future step (see eval/failure_modes.md).
"""

import argparse
import json
import sys
import time
from pathlib import Path

from run_session import run_session


def main() -> int:
    parser = argparse.ArgumentParser(description="campus-access benchmark")
    parser.add_argument("--video", required=True, help="Test video path")
    parser.add_argument("--ped-line", required=True, help="Pedestrian gate line")
    parser.add_argument("--veh-line", required=True, help="Vehicle gate line")
    parser.add_argument(
        "--ground-truth",
        default=None,
        help="Ground-truth JSON for accuracy comparison (optional)",
    )
    parser.add_argument(
        "--output",
        default="benchmark_report.json",
        help="Where to write the benchmark report",
    )

    args = parser.parse_args()

    ped_start, ped_end = _parse_line(args.ped_line)
    veh_start, veh_end = _parse_line(args.veh_line)

    print(f"Running campus-access on {args.video}...", file=sys.stderr)
    start = time.time()
    result = run_session(
        video_path=args.video,
        pedestrian_line=(ped_start, ped_end),
        vehicle_line=(veh_start, veh_end),
    )
    elapsed = time.time() - start

    fps_processed = result["n_frames"] / elapsed if elapsed > 0 else 0

    report = {
        "video": args.video,
        "n_frames": result["n_frames"],
        "elapsed_seconds": elapsed,
        "fps_processed": fps_processed,
        "pedestrian_zone": result["pedestrian_zone"],
        "vehicle_zone": result["vehicle_zone"],
        "dwell_summary": result["dwell_summary"],
        "ground_truth_comparison": (
            _compare_to_ground_truth(args.ground_truth, result)
            if args.ground_truth
            else None
        ),
    }

    Path(args.output).write_text(json.dumps(report, indent=2))
    print(f"Throughput: {fps_processed:.1f} fps", file=sys.stderr)
    print(f"Report written to {args.output}", file=sys.stderr)
    return 0


def _compare_to_ground_truth(gt_path: str, result: dict) -> dict:
    """Compare the session result to a hand-labeled ground truth.

    Implementation is a stub - real comparison requires a GT format that
    pairs frame indices with bboxes and IDs. See failure_modes.md for
    what ground-truth collection looks like in practice.
    """
    return {
        "status": "not_implemented",
        "note": "Ground-truth comparison requires hand-labeled data. See failure_modes.md.",
    }


def _parse_line(arg: str):
    parts = arg.split()
    a = tuple(int(v) for v in parts[0].split(","))
    b = tuple(int(v) for v in parts[1].split(","))
    return a, b


if __name__ == "__main__":
    sys.exit(main())
