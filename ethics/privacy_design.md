# Privacy Design for campus-access

## What this system does

`campus-access` counts people and vehicles entering and exiting a building, tracks how long they stay, and answers plain-English questions about the resulting data. It is designed for campus buildings: lecture halls, libraries, computer labs, parking gates.

## What this system does NOT do

This is the part that matters, so it gets its own section:

1. **No re-identification.** A person who enters a building is assigned a random integer (the "tracker ID") for that appearance. The ID is not reused after they leave. There is no database of tracker IDs. There is no way to map a tracker ID from today's session to a tracker ID from yesterday's session.

2. **No face recognition.** The detector finds boxes around people; it does not produce face embeddings, biometric templates, or anything that could be used to identify an individual across appearances.

3. **No persistence of per-person records.** Aggregated statistics are written to disk: total entries, total exits, dwell-time distributions. Per-tracker-ID records are not written. The output of the system is a summary, not a roster.

4. **No cloud calls by default.** The system runs on-device. Network calls are blocked at runtime by `pipeline/privacy.py::fail_if_cloud_call`; the only outbound connections allowed are loopback (for a local web UI, if one is added later) and link-local (for local network file sharing).

5. **No cross-camera correlation.** Each camera (or video file) is processed as a separate session. There is no mechanism to merge tracker IDs across cameras. Aggregating per-camera reports is a manual step that an operator does explicitly, with the documentation that this is an aggregate, not a re-identification.

## Why these guarantees matter

A campus access counter is the kind of system that, if deployed carelessly, becomes a surveillance system. The same hardware (a camera at a doorway) and the same software (a person detector) can answer two very different questions:

- "How many people used the library today?" — useful for staffing.
- "Which students skipped class this morning?" — surveillance.

The hardware is identical. The difference is in the data the system stores and exposes. `campus-access` is designed so that the second question is not just unanswered — it is not *askable*. There is no per-person record to query, so there is no way to produce the answer even if someone tried.

## How the guarantees are enforced

The privacy contract is enforced at three layers:

### Layer 1: Configuration

`PrivacyConfig` defaults to:

```python
PrivacyConfig(
    on_device_only=True,
    no_per_person_persistence=True,
    session_scoped_ids=True,
    refuse_face_recognition=True,
)
```

Loosening any of these requires an explicit environment variable:

| Variable | Effect |
|---|---|
| `CAMPUS_ACCESS_ON_DEVICE_ONLY=false` | Disables the network guard |
| `CAMPUS_ACCESS_NO_PERSON_PERSISTENCE=false` | Allows writing per-tracker-ID records |
| `CAMPUS_ACCESS_SESSION_SCOPED_IDS=false` | Allows ID reuse across sessions |
| `CAMPUS_ACCESS_REFUSE_FACE_RECOGNITION=false` | Allows face-recognition libraries |

The defaults are the privacy-strict options. There is no API call that bypasses them silently; the operator has to set the env var and restart the process.

### Layer 2: Code

`pipeline/privacy.py` implements `fail_if_cloud_call`, which monkey-patches `socket.socket.connect` to refuse non-loopback, non-link-local connections when `on_device_only=True`. Any code path that tries to reach the public internet raises `NetworkBlockedError`.

The tracker (`pipeline/tracker.py`) emits `TrackedObject` records that carry:

- A `tracker_id` (random integer, never reused across sessions).
- A class label (person/vehicle).
- First/last seen frames and a bbox.

There is no API on `TrackedObject` to attach a name, an ID number, or any other persistent identifier. Adding one would require changing the dataclass — there is no way to bolt it on through configuration alone.

### Layer 3: Output

The only thing that gets written to disk is:

- The annotated video (frames with detection boxes and line overlays). This contains faces if the camera captures faces; the operator is responsible for not deploying this in a context where that is a problem.
- A JSON summary with aggregated counts and dwell-time distributions.

There is no per-tracker-ID JSON, no CSV of who-came-when, no SQLite database of attendance. If you want those, you have to build them yourself outside this codebase.

## What this system is for

- Counting foot traffic at campus buildings for staffing decisions.
- Measuring library or lab usage for resource allocation.
- Generating aggregates ("how busy was it?") without exposing individuals.
- Building dashboards that respect privacy by construction.

## What this system is NOT for

- Tracking individual students or staff.
- Producing attendance lists for any purpose.
- Re-identifying people across cameras or sessions.
- Any application where per-person data is the goal.

If you need any of those, this is the wrong system. The privacy design is not a feature to disable; it is a property the system has by construction, and working around it would require modifying the source code in ways that should be obvious to anyone reviewing the change.

## License and contribution

This privacy design is part of the codebase. Changes to `pipeline/privacy.py`, `PrivacyConfig`, or this document should be reviewed with the same care as changes to authentication code in any other system.
