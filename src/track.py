# src/track.py

from deep_sort_realtime.deepsort_tracker import DeepSort

def initialize_tracker():
    tracker = DeepSort(
        max_age=30,
        n_init=3,
        max_cosine_distance=0.3,
        nn_budget=None,
        override_track_class=None
    )
    return tracker

def track_players(tracker, detections, frame):
    if len(detections) == 0:
        return []

    # Convert detections from [x, y, w, h, conf] to ([x1, y1, x2, y2], conf, class_name)
    converted_detections = []
    for det in detections:
        x, y, w, h, conf = det
        bbox = [x, y, x + w, y + h]
        converted_detections.append((bbox, conf, "person"))

    tracks = tracker.update_tracks(converted_detections, frame=frame)

    output = []
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, r, b = track.to_ltrb()
        output.append({
            'id': track_id,
            'bbox': [int(l), int(t), int(r), int(b)]
        })

    return output
