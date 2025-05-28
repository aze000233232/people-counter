import torch
import cv2
import numpy as np
import sys
import time
sys.path.append('sort')  
from sort import Sort

model = torch.hub.load('ultralytics/yolov5', 'yolov5l')
model.classes = [0]

video_path = "video/12.mp4"
cap = cv2.VideoCapture(video_path)

fps = 20
frame_time = 1.0 / fps

tracker = Sort()
unique_ids = set()
track_timers = {}  # track_id: first_seen_time
min_time = 2  # seconds a person must be tracked before counting

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()

    # Prepare detections for SORT: [x1, y1, x2, y2, score]
    dets = []
    for *xyxy, conf, cls in detections:
        dets.append([*xyxy, conf])
    dets = np.array(dets)
    if dets.shape[0] == 0:
        dets = np.empty((0, 5))

    # Update tracker
    tracks = tracker.update(dets)

    current_time = time.time()
    current_ids = set()

    # Draw and count unique IDs after 2 seconds
    for track in tracks:
        x1, y1, x2, y2, track_id = map(int, track)
        current_ids.add(track_id)
        if track_id not in track_timers:
            track_timers[track_id] = current_time
        # If tracked for at least min_time seconds, count as unique
        if (current_time - track_timers[track_id] >= min_time) and (track_id not in unique_ids):
            unique_ids.add(track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Remove timers for IDs no longer present
    for tid in list(track_timers.keys()):
        if tid not in current_ids and tid not in unique_ids:
            del track_timers[tid]

    cv2.putText(frame, f"Unique Entry Count: {len(unique_ids)}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.putText(frame, "Press 'q' to quit", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('People Detection (SORT Tracking)', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()