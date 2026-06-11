# Roboflow Pose Tracking

Real-time human pose tracking from a live camera feed using [Ultralytics YOLOv8 Pose](https://docs.ultralytics.com/tasks/pose/) for keypoint detection and [Supervision](https://supervision.roboflow.com/) for annotation.

Each frame from the webcam is run through a YOLOv8 pose model, and the detected skeleton edges and vertices are drawn back onto the frame and displayed in a window.

## Requirements

- Python >= 3.12
- A webcam
- [uv](https://docs.astral.sh/uv/) for dependency management

## Installation

Clone the repository and install dependencies into a virtual environment with `uv`:

```bash
uv sync
```

This installs the project dependencies declared in `pyproject.toml`:

- `supervision` — keypoint annotation utilities
- `ultralytics` — YOLOv8 pose model
- `opencv-python` — camera capture and display

> If OpenCV is not already pulled in transitively, add it explicitly:
>
> ```bash
> uv add opencv-python
> ```

The YOLOv8 pose weights (`yolov8m-pose.pt`) are downloaded automatically on first run.

## Usage

Run the app with `uv`:

```bash
uv run main.py
```

A window titled **Pose Tracking** opens showing the annotated camera feed. Press `q` to quit.

### Configuration

- **Camera source**: change the device index in `main.py` if you have multiple cameras:

```python
cap = cv2.VideoCapture(0)  # try 1, 2, ... for other cameras
```

- **Model**: swap `yolov8m-pose.pt` for a smaller/faster (`yolov8n-pose.pt`) or larger/more accurate (`yolov8x-pose.pt`) variant in `main.py`.

## How it works

1. `cv2.VideoCapture` reads frames from the webcam.
2. Each frame is passed to the YOLOv8 pose model, producing keypoints.
3. `sv.EdgeAnnotator` and `sv.VertexAnnotator` draw the skeleton onto the frame.
4. The annotated frame is shown with OpenCV until `q` is pressed.
