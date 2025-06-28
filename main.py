# main.py

from src.detect import detect_players
from src.track import initialize_tracker, track_players
from src.reid import ReIdentifier
from src.utils import draw_boxes, save_log
import cv2
import os
import datetime

# === Configuration ===
VIDEO_PATH = 'videos/15sec_input_720p.mp4'
MODEL_PATH = 'models/yolov11_player.pt'
DEBUG = False  # Set to True to show live debug window

# === Auto-named output paths based on timestamp ===
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_VIDEO_PATH = f'{OUTPUT_DIR}/output_{timestamp}.mp4'
LOG_PATH = f'{OUTPUT_DIR}/log_{timestamp}.csv'

# === Initialize Models ===
detector = detect_players(MODEL_PATH)
tracker = initialize_tracker()
reidentifier = ReIdentifier()

# === Video Setup ===
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

frame_idx = 0
log_data = []

# === Main Loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    print(f"Processing Frame {frame_idx}", end="\r")

    detections = detector(frame)

    # 🔴 Visualize raw detections (optional for debug)
    if DEBUG:
        for det in detections:
            x, y, w, h, conf = det
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, f"det:{conf:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 🟩 Track and Re-identify
    tracks = track_players(tracker, detections, frame)
    updated_tracks = reidentifier.update(tracks, frame)

    # 🟦 Draw final tracked IDs
    draw_boxes(frame, updated_tracks)

    # ✍️ Log tracking output
    log_data.extend([(frame_idx, t['id'], *t['bbox']) for t in updated_tracks])

    # 🎞️ Save frame to output video
    out.write(frame)

    # 👁️ Optional live preview
    if DEBUG:
        cv2.imshow("Detection Debug", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# === Cleanup ===
cap.release()
out.release()
if DEBUG:
    cv2.destroyAllWindows()

save_log(log_data, LOG_PATH)

# === Summary Output ===
print(f"\n✅ Processing complete.")
print(f"🎥 Video saved to: {OUTPUT_VIDEO_PATH}")
print(f"📄 Log saved to:   {LOG_PATH}")
