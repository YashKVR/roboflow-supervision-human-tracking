"""Real-time 3D motion capture.

Captures webcam frames, estimates 3D body landmarks with MediaPipe's
PoseLandmarker (metric `pose_world_landmarks`), and renders a live stick-figure
in a MuJoCo viewer where bones are drawn as capsules ("pipes") and joints as
spheres -- a lightweight mocap visualizer.

Run with:  uv run mocap3d.py
Press 'q' or ESC in the camera window (or close the viewer) to quit.
"""

from __future__ import annotations

import os
import time

import cv2
import mediapipe as mp
import mujoco
import mujoco.viewer
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_full.task")

# Bone connections for the 33-landmark pose model: list of (start, end) pairs.
POSE_CONNECTIONS = [
    (c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS
]

# --- Scene -----------------------------------------------------------------
# Minimal world: a floor + light. The skeleton itself is drawn every frame as
# user scene geometry, so no bodies/physics are needed.
_SCENE_XML = """
<mujoco model="mocap">
  <visual>
    <global offwidth="1280" offheight="720" azimuth="120" elevation="-15"/>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"
             width="512" height="3072"/>
    <texture type="2d" name="grid" builtin="checker" rgb1="0.2 0.3 0.4"
             rgb2="0.1 0.15 0.2" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="5 5 0.1" material="grid"/>
  </worldbody>
</mujoco>
"""

# Visual tuning ------------------------------------------------------------
SCALE = 1.0          # MediaPipe world units are already ~meters
HEIGHT_OFFSET = 1.0  # lift hips (the origin) above the floor
MIRROR = True        # mirror left/right so it feels like a mirror
BONE_RADIUS = 0.025
JOINT_RADIUS = 0.035
BONE_RGBA = np.array([0.9, 0.5, 0.1, 1.0], dtype=np.float32)
JOINT_RGBA = np.array([0.1, 0.7, 0.95, 1.0], dtype=np.float32)
MIN_VISIBILITY = 0.5


def landmark_to_mujoco(lm) -> np.ndarray:
    """Map a MediaPipe world landmark to MuJoCo world coords (z = up)."""
    x = -lm.x if MIRROR else lm.x
    return np.array(
        [
            x * SCALE,                       # right
            lm.z * SCALE,                    # depth (into screen)
            -lm.y * SCALE + HEIGHT_OFFSET,   # up
        ],
        dtype=np.float64,
    )


def _visible(lm) -> bool:
    vis = getattr(lm, "visibility", 1.0)
    return vis is None or vis >= MIN_VISIBILITY


def add_sphere(scene, pos: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, radius, radius], dtype=np.float64),
        pos,
        np.eye(3).flatten(),
        rgba,
    )
    scene.ngeom += 1


def add_capsule(scene, p1: np.ndarray, p2: np.ndarray, radius: float,
                rgba: np.ndarray) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).flatten(),
        rgba,
    )
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, p1, p2)
    scene.ngeom += 1


def draw_skeleton(scene, world_landmarks) -> None:
    """Populate the user scene with bones (capsules) and joints (spheres)."""
    scene.ngeom = 0
    points = [landmark_to_mujoco(lm) for lm in world_landmarks]

    for start_idx, end_idx in POSE_CONNECTIONS:
        if not (_visible(world_landmarks[start_idx])
                and _visible(world_landmarks[end_idx])):
            continue
        add_capsule(scene, points[start_idx], points[end_idx], BONE_RADIUS,
                    BONE_RGBA)

    for idx, lm in enumerate(world_landmarks):
        if _visible(lm):
            add_sphere(scene, points[idx], JOINT_RADIUS, JOINT_RGBA)


def draw_overlay(frame: np.ndarray, image_landmarks) -> None:
    """Draw the 2D skeleton on the camera frame for reference."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in image_landmarks]
    for start_idx, end_idx in POSE_CONNECTIONS:
        cv2.line(frame, pts[start_idx], pts[end_idx], (245, 117, 66), 2)
    for p in pts:
        cv2.circle(frame, p, 4, (245, 200, 66), -1)


def build_landmarker() -> vision.PoseLandmarker:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Pose model not found at {MODEL_PATH}. Download it with:\n"
            "  curl -L -o pose_landmarker_full.task "
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
        )
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera feed")

    model = mujoco.MjModel.from_xml_string(_SCENE_XML)
    data = mujoco.MjData(model)
    start = time.monotonic()

    with build_landmarker() as landmarker, \
            mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -15
        viewer.cam.lookat[:] = [0.0, 0.0, 1.0]

        while viewer.is_running():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_world_landmarks:
                draw_skeleton(viewer.user_scn, result.pose_world_landmarks[0])
                viewer.sync()
            if result.pose_landmarks:
                draw_overlay(frame, result.pose_landmarks[0])

            cv2.imshow("MediaPipe Pose (camera)", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
