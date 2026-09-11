# Failure modes for campus-access

This document records the cases where the system is known to underperform, fail, or produce wrong answers. It exists so that an operator deploying the system can predict when to expect degraded output.

## Detection failures

### 1. Heavy occlusion

When two or more people walk through the doorway simultaneously and overlap in the camera's view, the detector may merge them into a single detection or miss one entirely. ByteTrack then assigns the merged detection a single ID, so two people are tracked as one.

**Mitigation:** none at the algorithm level. The mitigation is operational: position the camera so the doorway is wide enough that two people can pass with visible separation between them.

### 2. Backlight / silhouettes

Cameras pointed toward a bright outdoor scene produce silhouetted people whose features are not visible. YOLO detects silhouettes, but with lower confidence and at a higher false-positive rate.

**Mitigation:** adjust camera exposure or position; alternatively, lower the confidence threshold at the cost of more false positives.

### 3. Far-field objects

People in the back of a deep scene are detected less reliably than people in the foreground. At sufficient distance, the per-frame detection may be intermittent, causing ByteTrack to lose the ID and reassign a new one when the detection comes back.

**Mitigation:** crop the frame to the area of interest (the doorway) before processing, or use a higher-resolution camera.

## Tracking failures

### 1. ID switches

If the detector briefly loses an object (one missed frame) and then re-detects it, ByteTrack may assign a new ID instead of recovering the old one. This manifests as a single physical person appearing as two distinct tracker IDs.

**Mitigation:** ByteTrack's `minimum_consecutive_frames` parameter controls how many missed frames are tolerated before the track is dropped. The default is 1, which is the strictest setting; raising it makes recovery more forgiving at the cost of more false-positive tracks.

### 2. Identity drift for vehicles

A car parked in front of the camera for several seconds may have its ID reassigned if it moves slightly between detections (e.g., the driver shifts the car). This is the same failure mode as ID switches but is more common for vehicles because they typically appear in fewer frames per unit time than pedestrians (cars move faster through the field of view).

## Counting failures

### 1. Line zone placement

The line zone has to be placed so that objects cross it cleanly. If the line is placed too close to the edge of the frame, objects may enter and exit without crossing it, and the crossing count will be wrong.

**Mitigation:** position the line so that any object that appears in the frame must cross it at some point.

### 2. Multiple people crossing at once

If two people cross the line in the same frame, supervision's LineZone will record one crossing event, not two. The system's count will be lower than the true count.

**Mitigation:** none at the algorithm level. Operational mitigation: position the doorway so two people cross at slightly different times.

## Dwell-time failures

### 1. Dwell time for parked objects

A car parked in view for 30 minutes will be tracked for 30 minutes, which is correct, but the system will not record a crossing-out event because the car never leaves. The dwell record stays open until session end.

**Mitigation:** `DwellTimeAnalyzer.finalize()` reports `still_present` count separately from closed records, so the operator can see how many objects were never observed leaving.

### 2. Dwell time for stationary people

A person standing in a doorway without moving will be tracked continuously but will not cross the line. Their dwell record stays open until they leave the frame.

**Mitigation:** same as above; `still_present` makes this visible in the summary.

## Query interface failures

### 1. Ambiguous questions

The query parser is rule-based and recognizes a fixed set of question shapes. Questions outside that set return `QueryType.UNKNOWN` and the responder says "I don't understand that question."

Example: "How does the foot traffic at the engineering building compare to the library?" — this requires cross-camera aggregation, which the parser does not attempt.

**Mitigation:** rewrite the question to a shape the parser recognizes. See `query/nl_interface.py` for the keyword sets.

### 2. Time window ambiguity

"Yesterday" is interpreted as the previous calendar day relative to the system clock, not relative to the video's timestamp. If the video is from a week ago, "yesterday" returns data from the wrong day.

**Mitigation:** the parser does not attempt to disambiguate timestamps relative to video metadata. The operator must frame queries in terms of the session being analyzed, not relative to the current date.

## Privacy failure modes

### 1. Annotated video leaks faces

The annotated output video shows detection boxes drawn around people. If the camera captures faces (and most do), the output video contains faces.

**Mitigation:** do not distribute the annotated video publicly. If face-anonymized output is required, redact the frames before writing.

### 2. Tracker ID reuse across sessions is impossible by construction

There is no configuration that allows tracker IDs to be reused across sessions. This is not a failure mode; it is a property. Operators who want to track individuals across sessions must use a different system.

## What the benchmark does not measure

- **Identity persistence** across occlusion: not measured. The benchmark tracks `n_frames` and throughput but does not compare tracker IDs to ground-truth identities.
- **Re-identification accuracy**: by design, not measured. The system does not re-identify.
- **Real-world lighting variation**: the demo videos are shot in controlled conditions. Real campus cameras have varied lighting (HVAC cycles, automatic exposure flicker). A production deployment should be tested in the actual deployment environment.

## What to do when the system fails

The benchmark reports throughput and counts. If the counts disagree with a manual count for a session, the most common cause is one of the above detection or counting failures. The annotated output video is the first place to look for diagnosis: the line zone placement and detection boxes show what the system actually saw.
