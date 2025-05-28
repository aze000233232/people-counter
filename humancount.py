import torch
import cv2
import numpy as np
import sys
sys.path.append('sort')  # Hanapin nyo ung path ng sort file
from sort import Sort

model = torch.hub.load('ultralytics/yolov5', 'yolov5l')
model.classes = [0]  

cap = cv2.VideoCapture(0)  

fps = 20
frame_time = 1.0 / fps

tracker = Sort()
unique_ids = set()

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

    # Draw and count unique IDs
    for track in tracks:
        x1, y1, x2, y2, track_id = map(int, track)
        unique_ids.add(track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, f"Unique Entry Count: {len(unique_ids)}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.putText(frame, f"People Now: {len(tracks)}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, "Press 'q' to quit, 's' to save & quit", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Live People Detection', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):
        with open("people_count.txt", "w") as f:
            f.write(f"Unique Entry Count: {len(unique_ids)}\n")
        print(f"People count saved to people_count.txt: {len(unique_ids)}")
        break

cap.release()
cv2.destroyAllWindows()