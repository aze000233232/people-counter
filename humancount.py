import torch
import cv2
import numpy as np
import sys
import time
sys.path.append('sort')  # Hanapin nyo ung path ng sort file
from sort import Sort


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.classes = [0]  


cap = cv2.VideoCapture(0)
fps = 20
frame_time = 1.0 / fps

tracker = Sort()
unique_ids = set()


button_coords = {
    "SAVE": (0, 0, 0, 0),
    "QUIT": (0, 0, 0, 0)
}
clicked_button = None

def mouse_callback(event, x, y, flags, param):
    global clicked_button
    if event == cv2.EVENT_LBUTTONDOWN:
        for name, (x1, y1, x2, y2) in button_coords.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                clicked_button = name

cv2.namedWindow("Live People Detection UI")
cv2.setMouseCallback("Live People Detection UI", mouse_callback)

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 360))
    frame = cv2.flip(frame, 1)

    
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()

    
    dets = []
    for *xyxy, conf, cls in detections:
        dets.append([*xyxy, conf])
    dets = np.array(dets)
    if dets.shape[0] == 0:
        dets = np.empty((0, 5))

    
    tracks = tracker.update(dets)

    for track in tracks:
        x1, y1, x2, y2, track_id = map(int, track)
        unique_ids.add(track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    
    canvas = np.zeros((550, 1000, 3), dtype=np.uint8)
    webcam_top_left = (20, 20)
    canvas[webcam_top_left[1]:webcam_top_left[1]+360,
           webcam_top_left[0]:webcam_top_left[0]+640] = frame
    cv2.rectangle(canvas, webcam_top_left, (webcam_top_left[0]+640, webcam_top_left[1]+360), (200, 0, 255), 3)

    
    cv2.rectangle(canvas, (700, 20), (950, 170), (100, 100, 255), 3)
    cv2.putText(canvas, f"Entry Count", (740, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(canvas, f"{len(unique_ids)}", (830, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    
    cv2.rectangle(canvas, (700, 190), (950, 340), (100, 100, 255), 3)
    cv2.putText(canvas, f"People Now", (735, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(canvas, f"{len(tracks)}", (830, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    
    canvas_width = canvas.shape[1]
    button_w, button_h = 200, 60
    y_button = 470

    x1_save = int(canvas_width / 2 - 220)
    y1_save = y_button
    x2_save = x1_save + button_w
    y2_save = y1_save + button_h
    button_coords["SAVE"] = (x1_save, y1_save, x2_save, y2_save)
    cv2.rectangle(canvas, (x1_save, y1_save), (x2_save, y2_save), (128, 0, 128), 2)
    cv2.putText(canvas, "SAVE", (x1_save + 50, y1_save + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    x1_quit = int(canvas_width / 2 + 20)
    y1_quit = y_button
    x2_quit = x1_quit + button_w
    y2_quit = y1_quit + button_h
    button_coords["QUIT"] = (x1_quit, y1_quit, x2_quit, y2_quit)
    cv2.rectangle(canvas, (x1_quit, y1_quit), (x2_quit, y2_quit), (128, 0, 128), 2)
    cv2.putText(canvas, "QUIT", (x1_quit + 50, y1_quit + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    
    cv2.imshow("Live People Detection UI", canvas)
    key = cv2.waitKey(1) & 0xFF

   
    if key == ord('q') or clicked_button == "QUIT":
        print("Program exited.")
        break
    if key == ord('s') or clicked_button == "SAVE":
        with open("people_count.txt", "w") as f:
            f.write(f"Unique Entry Count: {len(unique_ids)}\n")
        print(f"People count saved to people_count.txt: {len(unique_ids)}")
        break

    clicked_button = None

    
    elapsed = time.time() - start_time
    if elapsed < frame_time:
        time.sleep(frame_time - elapsed)


cap.release()
cv2.destroyAllWindows()
