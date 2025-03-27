import cv2
import numpy as np
import math
import time
from ultralytics import YOLO

# Function to mask out the region of interest
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)

# Function to draw lane lines
def draw_lane_lines(img, left_line, right_line, color=(0, 255, 0)):
    overlay = np.zeros_like(img)
    poly_pts = np.array([[ 
        (left_line[0], left_line[1]),
        (left_line[2], left_line[3]),
        (right_line[2], right_line[3]),
        (right_line[0], right_line[1])
    ]], dtype=np.int32)
    cv2.fillPoly(overlay, poly_pts, color)
    return cv2.addWeighted(img, 0.8, overlay, 0.5, 0.0)

# Lane detection pipeline
def pipeline(image):
    height, width = image.shape[:2]
    roi_vertices = [(0, height), (width // 2, height // 2), (width, height)]

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    masked_edges = region_of_interest(edges, np.array([roi_vertices], np.int32))
    lines = cv2.HoughLinesP(masked_edges, 2, np.pi / 180, 160, minLineLength=40, maxLineGap=25)

    if lines is None:
        return image

    left_x, left_y, right_x, right_y = [], [], [], []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
        if abs(slope) < 0.5:
            continue
        if slope < 0:
            left_x += [x1, x2]
            left_y += [y1, y2]
        else:
            right_x += [x1, x2]
            right_y += [y1, y2]

    min_y, max_y = int(height * 3 / 5), height
    left_x_start, left_x_end, right_x_start, right_x_end = 0, 0, 0, 0

    try:
        if left_x and left_y:
            left_fit = np.polyfit(left_y, left_x, 1)
            left_x_start, left_x_end = int(left_fit[0] * max_y + left_fit[1]), int(left_fit[0] * min_y + left_fit[1])
        if right_x and right_y:
            right_fit = np.polyfit(right_y, right_x, 1)
            right_x_start, right_x_end = int(right_fit[0] * max_y + right_fit[1]), int(right_fit[0] * min_y + right_fit[1])
    except:
        return image

    return draw_lane_lines(image, [left_x_start, max_y, left_x_end, min_y], [right_x_start, max_y, right_x_end, min_y])

# Function to estimate distance based on bounding box size
def estimate_distance(bbox_width, focal_length=1000, known_width=2.0):
    return (known_width * focal_length) / bbox_width

# Main function to process video with YOLO and lane detection
def process_video():
    model = YOLO('weights/yolov8n.pt')
    cap = cv2.VideoCapture('video/vdo1.mp4')

    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('video/output.mp4', fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (frame_width, frame_height))
        lane_frame = pipeline(frame)

        results = model(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf, cls = box.conf[0], int(box.cls[0])

                if model.names[cls] == 'car' and conf >= 0.6:  
                    cv2.rectangle(lane_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    label = f'{model.names[cls]} {conf:.2f}'
                    cv2.putText(lane_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    distance = estimate_distance(x2 - x1)
                    cv2.putText(lane_frame, f'Distance: {distance:.2f}m', (x1, y2 + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        out.write(lane_frame)
        cv2.imshow('Lane and Car Detection', lane_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Processing complete! Video saved as 'video/output.mp4'")

process_video()
