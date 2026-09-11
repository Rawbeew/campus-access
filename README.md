# campus-access

**Privacy-preserving entry/exit counter for campus buildings. Tracks pedestrians and vehicles separately. Records dwell time. Queryable in English.**

## What it does

`campus-access` answers three questions about any campus building entrance:

1. **How many people/vehicles came in?** — Counts entries and exits at two separate line zones (one for pedestrian gates, one for vehicle gates).
2. **How long did they stay?** — Tracks dwell time per tracked object from first-seen to last-seen.
3. **What was the pattern?** — Answers plain-English questions like "how busy was it this morning?" or "compare Monday vs Friday traffic".

It does not answer: "who came in?", "which student was absent?", or any question that requires per-person records. Those records are not stored. See [`ethics/privacy_design.md`](ethics/privacy_design.md) for the privacy contract.

## Pipeline

```
Video --> YOLO detects people & vehicles
      --> ByteTrack assigns session-scoped IDs
      --> Two line zones (pedestrian gate + vehicle gate) record crossings
      --> Dwell analyzer aggregates first-seen / last-seen per ID
      --> Output: annotated video + JSON summary (no per-person data)
```

| Component | What it does |
|---|---|
| `pipeline/detector.py` | YOLO wrapper. Restricts to campus-relevant classes (person, bicycle, car, motorcycle, bus). |
| `pipeline/tracker.py` | Two parallel ByteTrack instances (one for pedestrians, one for vehicles). Disjoint ID spaces. |
| `pipeline/line_zones.py` | Virtual lines at pedestrian and vehicle gates. Records inward/outward crossings with class metadata. |
| `pipeline/dwell_time.py` | First-seen / last-seen per tracked ID, in real seconds. Closed when object exits frame or crosses outward. |
| `pipeline/privacy.py` | Network guard, configuration defaults, and the policy enforcement layer. |
| `query/nl_interface.py` | Rule-based English parser. Recognizes a fixed set of question shapes; fails closed on unknowns. |

## Stack

- **YOLOv8** (ultralytics) - object detection
- **ByteTrack** - object tracking via the `supervision` wrapper
- **supervision** - annotations, line zones, byte trackers
- **OpenCV** - video I/O
- **Python 3.10+** - no other runtime dependencies

## How to run

### Command line

```bash
# Install dependencies (one-time)
pip install ultralytics supervision opencv-python numpy

# Run a session
python run_session.py \
  --video sample_input_vehicle.mp4 \
  --ped-line "0,540 1920,540" \
  --veh-line "0,540 1920,540" \
  --output-video annotated.mp4 \
  --output-summary session.json \
  --yolo-model yolov8n.pt
```

The line arguments are in pixel coordinates relative to the input video. For a 1920x1080 frame, `0,540 1920,540` is a horizontal line across the middle. Adjust for your camera angle.

### Python API

```python
from run_session import run_session

result = run_session(
    video_path="sample_input_vehicle.mp4",
    pedestrian_line=((0, 540), (1920, 540)),
    vehicle_line=((0, 540), (1920, 540)),
    output_video_path="annotated.mp4",
    output_summary_path="session.json",
)
print(result["dwell_summary"])
```

### Asking questions after a session

```python
from query.nl_interface import QueryParser, QueryResponder

parser = QueryParser()
responder = QueryResponder()

q = parser.parse("How many people came in this morning?")
print(responder.answer(q, summary))
# -> "47 pedestrians and 6 vehicles were tracked in this session..."

q = parser.parse("Compare Monday vs Friday traffic")
print(responder.answer(q, summary))
# -> "Comparison requested between Monday and Friday. This session's summary..."
```

## Privacy

The privacy contract is the headline feature, not an afterthought. Read [`ethics/privacy_design.md`](ethics/privacy_design.md) before deploying.

Short version:

- **No re-identification.** Tracker IDs are session-scoped integers, never reused, never persisted.
- **No face recognition.** Face-recognition libraries refuse to import if `refuse_face_recognition=True`.
- **No per-person records on disk.** Aggregates only.
- **No cloud calls.** The system runs on-device. Outbound HTTP raises `NetworkBlockedError`.

## Demo footage

This repo includes two short sample videos, both from Pexels:

| File | Content |
|---|---|
| `sample_input_pedestrian.mp4` | People entering/exiting a building (good for testing the pedestrian line zone) |
| `sample_input_vehicle.mp4` | Cars driving past a fixed camera (good for testing the vehicle line zone) |

## What this is for

- Counting foot traffic at campus buildings for staffing decisions.
- Measuring library or lab usage for resource allocation.
- Building "how busy was it?" dashboards that respect privacy by construction.
- Research on campus operations, learning analytics, or building management.

## What this is NOT for

- Tracking individual students or staff.
- Producing attendance lists.
- Re-identifying people across cameras or sessions.
- Any application where per-person data is the goal.

For those, you need a different system — one that is not `campus-access`.

## Cross-link

The same author builds computational stylometry tools for authorship attribution; see [nigerian-forensic-stylistics](https://github.com/Rawbeew/nigerian-forensic-stylistics) for the text-analysis counterpart to this computer-vision project.

## License

MIT, same as the underlying libraries.
