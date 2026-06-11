# Roboflow Pose Tracking

This project has two parts:

1. **2D pose tracking** (`main.py`) — real-time human pose overlay on a live camera feed using [Ultralytics YOLOv8 Pose](https://docs.ultralytics.com/tasks/pose/) for keypoint detection and [Supervision](https://supervision.roboflow.com/) for annotation.
2. **3D motion capture** (`mocap3d.py`) — a live 3D stick-figure rendered in a [MuJoCo](https://mujoco.readthedocs.io/) viewer, driven by [MediaPipe Pose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker), which provides true metric 3D landmarks (with depth). Bones are drawn as capsules ("pipes") and joints as spheres, mirroring your movement like a mocap system.

### Why two pose models?

YOLOv8 Pose outputs only **2D** keypoints `(x, y, confidence)` — no depth. For a real 3D figure we use MediaPipe's `PoseLandmarker`, whose `pose_world_landmarks` are metric 3D coordinates `(x, y, z)` in meters with the origin at the hips.

## Requirements

- Python >= 3.12
- A webcam
- A display (the MuJoCo viewer and OpenCV windows are GUI apps)
- [uv](https://docs.astral.sh/uv/) for dependency management

## Installation

Clone the repository and install dependencies into a virtual environment with `uv`:

```bash
uv sync
```

Dependencies (from `pyproject.toml`):

- `ultralytics` — YOLOv8 pose model (2D)
- `supervision` — keypoint annotation utilities (2D)
- `mediapipe` — 3D pose landmarks (mocap)
- `mujoco` — 3D viewer / skeleton rendering

> If OpenCV is not pulled in transitively, add it explicitly: `uv add opencv-python`

### Model files

- YOLOv8 weights (`yolov8m-pose.pt`) download automatically on first run of `main.py`.
- The MediaPipe pose model must be downloaded once for `mocap3d.py`:

```bash
curl -L -o pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

Both model files are git-ignored.

## Usage

### 2D pose overlay

```bash
uv run main.py
```

A window titled **Pose Tracking** opens showing the annotated camera feed. Press `q` to quit.

### 3D motion capture

```bash
uv run mocap3d.py
```

Two windows open: the **MuJoCo** 3D viewer with your live skeleton, and a camera window showing the 2D overlay. Press `q` or `ESC` in the camera window (or close the viewer) to quit. Drag in the MuJoCo window to orbit/zoom the camera.

## Configuration

### `main.py`

- **Camera source**: change the device index if you have multiple cameras:

```python
cap = cv2.VideoCapture(0)  # try 1, 2, ... for other cameras
```

- **Model**: swap `yolov8m-pose.pt` for a smaller/faster (`yolov8n-pose.pt`) or larger/more accurate (`yolov8x-pose.pt`) variant.

### `mocap3d.py`

Tunable constants near the top of the file:

- `MIRROR` — flip left/right so the figure feels like a mirror.
- `HEIGHT_OFFSET` — how far the hips (the origin) sit above the floor.
- `BONE_RADIUS` / `JOINT_RADIUS` — thickness of the pipes and joint spheres.
- `MIN_VISIBILITY` — hide low-confidence landmarks.
- `SCALE` — overall size multiplier.

The MediaPipe → MuJoCo coordinate mapping (MuJoCo is z-up) lives in `landmark_to_mujoco`.

## How it works

### 2D (`main.py`)

1. `cv2.VideoCapture` reads frames from the webcam.
2. Each frame is passed to the YOLOv8 pose model, producing keypoints.
3. `sv.EdgeAnnotator` and `sv.VertexAnnotator` draw the skeleton onto the frame.
4. The annotated frame is shown with OpenCV until `q` is pressed.

### 3D (`mocap3d.py`)

1. `cv2.VideoCapture` reads webcam frames.
2. MediaPipe `PoseLandmarker` (VIDEO mode) returns metric 3D `pose_world_landmarks`.
3. Landmarks are mapped into MuJoCo's z-up world frame.
4. Each frame, the viewer's user scene is rebuilt: a capsule per bone connection and a sphere per joint, drawn with `mjv_initGeom` / `mjv_connector`. No physics — just live rendering.
