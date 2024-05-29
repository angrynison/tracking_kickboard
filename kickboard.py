import torch
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import pathlib
import sys

pathlib.PosixPath = pathlib.WindowsPath

model_path = 'weight.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

video_path = 'kickboard2.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error reading video file")
    sys.exit()

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

line_points = [(1080, 0), (1080, 920)]
video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

object_tracker = DeepSort(
    max_age=5,
    n_init=2,
    nms_max_overlap=1.0,
    max_cosine_distance=0.3,
    nn_budget=None,
    override_track_class=None,
    embedder="mobilenet",
    half=True,
    bgr=True,
    embedder_gpu=True,
    embedder_model_name=None,
    embedder_wts=None,
    polygon=False,
    today=None
)

helmet_count = 0
no_helmet_count = 0
crossed_ids = set()

def is_crossing_line(x, prev_x, line_x):
    return (prev_x < line_x and x >= line_x) or (prev_x > line_x and x <= line_x)

prev_centers = {}

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    results = model(frame)
    detections = []

    for *box, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, box)
        detections.append([[x1, y1, x2 - x1, y2 - y1], conf, cls])

    tracker_outputs = object_tracker.update_tracks(detections, frame=frame)

    for track in tracker_outputs:
        bbox = track.to_ltrb()
        tid = track.track_id
        cls_id = int(track.det_class)
        cls_name = model.names[cls_id]
        center_x = int((bbox[0] + bbox[2]) / 2)
        center_y = int((bbox[1] + bbox[3]) / 2)

        if tid in prev_centers:
            prev_center_x = prev_centers[tid][0]
            if is_crossing_line(center_x, prev_center_x, line_points[0][0]) and tid not in crossed_ids:
                if cls_id == 0:  # Helmet
                    helmet_count += 1
                elif cls_id == 1:  # No Helmet
                    no_helmet_count += 1
                crossed_ids.add(tid)

        prev_centers[tid] = (center_x, center_y)
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
        cv2.putText(frame, f"{cls_name} {tid}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 0, 0), 2)

    cv2.line(frame, line_points[0], line_points[1], (0, 255, 255), 2)
    cv2.putText(frame, f"Helmet Count: {helmet_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"No Helmet Count: {no_helmet_count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    video_writer.write(frame)
    cv2.imshow('YOLOv5 DeepSORT Object Counting', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()

with open('object_count_log.txt', 'w') as log_file:
    log_file.write(f"Helmet Count: {helmet_count}\n")
    log_file.write(f"No Helmet Count: {no_helmet_count}\n")

print("Counting completed and log file saved.")
