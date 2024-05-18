import sys
import torch
import cv2
import random
import time
from ultralytics import YOLO
import utils.draw as draw

def yolov8_detection(model, image, q):
    img_size = 640
    confidence = 0.7
    stream_buffer = True
    verb = False
    
    if q == 'fp16':
        results = model.predict(image, imgsz=img_size, conf=confidence, verbose=verb, half=True)
    elif q == 'int8':
        results = model.predict(image, imgsz=img_size, conf=confidence, verbose=verb, int8=True)
    else:
        results = model.predict(image, imgsz=img_size, conf=confidence, verbose=verb)
    
    result = results[0].cpu()

    # Get information from result
    box = result.boxes.xyxy.numpy()
    conf = result.boxes.conf.numpy()
    cls = result.boxes.cls.numpy().astype(int)

    return cls, conf, box


def detect_camera(model_path, q):
    model = YOLO(model_path, task='detect')

    # Class Name and Colors
    label_map = model.names
    COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in label_map]
    
    # FPS
    frame_count = 0
    total_fps = 0
    avg_fps = 0
    recent_fps_list = []
    recent_fps_count = 10

    # Open camera
    video_cap = cv2.VideoCapture(0)

    while True:
        ret, frame = video_cap.read()
        if not ret:
            break

        # Start Time
        start = time.time()

        # Detection
        cls, conf, box = yolov8_detection(model, frame, q)

        # Pack together for easy use
        detection_output = list(zip(cls, conf, box))
        image_output = draw.box(frame, detection_output, label_map, COLORS)

        # End Time
        end = time.time()

        # Draw FPS
        frame_count += 1
        fps = 1 / (end - start)
        total_fps += fps
        recent_fps_list.append(fps)
        if len(recent_fps_list) > recent_fps_count:
            total_fps -= recent_fps_list.pop(0)
        avg_fps = total_fps / len(recent_fps_list)

        image_output = draw.fps(avg_fps, image_output)

        cv2.imshow('Detection', image_output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release camera and close windows
    video_cap.release()
    cv2.destroyAllWindows()