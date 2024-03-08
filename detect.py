import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import matplotlib.pyplot as plt

SOURCE = np.array([[1252, 787], [2298, 803], 
                   [5039, 2159], [-550, 2159]])

TARGET_WIDTH = 25
TARGET_HEIGHT = 250

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)

video_info = sv.VideoInfo.from_video_path("vehicles.mp4")

model = YOLO("yolov8x.pt")
tracker = sv.ByteTrack(frame_rate=video_info.fps)
frame_generator = sv.get_video_frames_generator("vehicles.mp4")
label_annotator = sv.LabelAnnotator()
bounding_box_annotator = sv.BoundingBoxAnnotator()
polygon_zone = sv.PolygonZone(SOURCE, frame_resolution_wh=video_info.resolution_wh)

class TransformationView:
    def __init__(self):
        source = SOURCE.astype(np.float32)
        target = TARGET.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)
        self.im = cv2.getPerspectiveTransform(target, source)
    
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)
    
tv = TransformationView()

for frame in frame_generator:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[polygon_zone.trigger(detections)]
    detections = tracker.update_with_detections(detections)

    points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)

    transformed_points = tv.transform_points(points)

    for tracker_id, point in zip(detections.tracker_id, transformed_points):
        print(tracker_id, point)

    break 