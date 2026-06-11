import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolov8m-pose.pt")
edge_annotator = sv.EdgeAnnotator()
vertex_annotator = sv.VertexAnnotator()

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame)[0]
    key_points = sv.KeyPoints.from_ultralytics(results)

    annotated_frame = edge_annotator.annotate(
        frame.copy(), key_points=key_points)
    return vertex_annotator.annotate(
        annotated_frame, key_points=key_points)

def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera feed")

    frame_index = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame = callback(frame, frame_index)
            frame_index += 1

            cv2.imshow("Pose Tracking", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
