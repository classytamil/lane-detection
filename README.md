# Lane and Vehicle Detection using YOLOv8

## Overview
This project implements real-time lane and vehicle detection using the YOLOv8 model and OpenCV. It processes video frames to detect lanes using edge detection and Hough transform while simultaneously detecting vehicles and estimating their distance using YOLOv8.

## Features
- **Lane detection** using edge detection and Hough Line Transform.
- **Vehicle detection** using YOLOv8.
- **Distance estimation** of detected vehicles.
- **Processes a video file** and saves the output with detected lanes and vehicles.

## Requirements
### Dependencies
Ensure you have the following dependencies installed:

```bash
pip install opencv-python numpy ultralytics
```

### Required Files
- YOLOv8 weights file: `yolov8n.pt`
- Input video file: `vdo.mp4`

## Installation & Setup
1. Clone the repository or copy the script.
2. Install the required dependencies.
3. Place the required files in the specified directory.
4. Run the script:

```bash
python lane_vehicle_detection.py
```

## Configuration
The following parameters can be adjusted based on camera calibration and input data:
- **KNOWN_WIDTH**: Approximate width of a car in meters.
- **FOCAL_LENGTH**: Estimated focal length of the camera.
- **VEHICLE_CLASSES**: List of vehicle classes to detect.

## How It Works
### 1. Lane Detection
- Converts frame to grayscale.
- Applies Gaussian blur.
- Uses Canny edge detection.
- Masks the region of interest.
- Detects lanes using Hough Line Transform.

### 2. Vehicle Detection & Distance Estimation
- Uses YOLOv8 for vehicle detection.
- Estimates distance using the formula:
  
  ```math
  Distance = \frac{Known Width \times Focal Length}{Bounding Box Width}
  ```

- Annotates frames with bounding boxes and distance labels.

### 3. Video Processing
- Reads the input video frame by frame.
- Processes each frame and writes to an output file.
- Displays real-time detection with an option to quit (`q`).

## Output
- The processed video with lane and vehicle detection is saved as `output.avi`.
- Detected vehicles are annotated with their class, confidence score, and estimated distance.
