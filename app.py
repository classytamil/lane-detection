import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO(r'E:\krishtech\projects\Lane_Detection\weights\yolov8n.pt')

# Define known parameters for distance estimation
KNOWN_WIDTH = 2.0  # Approximate car width in meters
FOCAL_LENGTH = 800  # Adjust this based on camera calibration

# List of vehicle classes in COCO dataset
VEHICLE_CLASSES = ["car", "bus", "truck", "motorcycle"]

def estimate_distance(bbox_width, focal_length=FOCAL_LENGTH, known_width=KNOWN_WIDTH):
    """Estimate the distance of a detected vehicle."""
    if bbox_width > 0:
        return (known_width * focal_length) / bbox_width
    return 0

def process_frame(frame, model):
    """Detect lanes and vehicles, then annotate the frame."""
    height, width = frame.shape[:2]

    # Define region of interest (ROI) for lane detection
    roi_vertices = [(0, height), (width // 2, height // 2), (width, height)]
    
    # Edge detection and lane detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Mask the ROI
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, np.array([roi_vertices], np.int32), 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Hough Line Transform for lane detection
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=25)
    
    # Draw detected lanes
    lane_overlay = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lane_overlay, (x1, y1), (x2, y2), (0, 255, 0), 5)
    frame = cv2.addWeighted(frame, 0.8, lane_overlay, 1, 1)

    # Object detection with YOLO
    results = model(frame)
    
    for result in results:
        boxes = result.boxes.xyxy
        classes = result.boxes.cls
        confidences = result.boxes.conf

        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = map(int, box)
            class_name = model.names[int(cls)]

            if class_name in VEHICLE_CLASSES and conf >= 0.5:
                bbox_width = x2 - x1
                distance = estimate_distance(bbox_width)
                
                # Display bounding box and distance
                label = f'{class_name} {conf:.2f} Dist: {distance:.2f}m'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return frame

# Open video file
cap = cv2.VideoCapture(r'E:\krishtech\projects\Lane_Detection\video\vdo.mp4')

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Set up output video writer
output_path = r'E:\krishtech\projects\Lane_Detection\output.avi'
output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = process_frame(frame, model)
    output_video.write(processed_frame)  # Save frame to output video

    # Display the processed frame
    cv2.imshow('Lane & Vehicle Detection', processed_frame)

    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
output_video.release()
cv2.destroyAllWindows()

print(f"Processed video saved at: {output_path}")
