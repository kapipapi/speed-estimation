import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import matplotlib.pyplot as plt

SAVE = False

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
label_annotator = sv.LabelAnnotator(text_scale=2, text_thickness=2)
bounding_box_annotator = sv.BoundingBoxAnnotator()
trace_annotator = sv.TraceAnnotator(
        trace_length=video_info.fps * 2,
        position=sv.Position.BOTTOM_CENTER,
    )
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

trackers = {}
trackers_speed = {}

plt.ion()
fig = plt.figure(figsize=(1, 5))

with sv.VideoSink("vehicles_tracked.mp4", video_info) as sink:
    for frame_id, frame in enumerate(frame_generator):
        results = model(frame, device='mps')[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[polygon_zone.trigger(detections)]
        detections = tracker.update_with_detections(detections)


        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        transformed_points = tv.transform_points(points)

        for tracker_id, point in zip(detections.tracker_id, transformed_points):
            if tracker_id in trackers:
                trackers[tracker_id] = np.concatenate((trackers[tracker_id], [point]))
            else:
                trackers[tracker_id] = np.array([point])
        
        plt.clf()
        speed_labels = []
        for tracker_id in detections.tracker_id:
            points = trackers[tracker_id]
            plt.scatter(points[:, 0], points[:, 1], label=tracker_id)

            time_window = 10
            
            if len(points) > time_window:
                distance = np.linalg.norm(points[-1] - points[-time_window]) 
                time = time_window / video_info.fps # meters per seconds 
                speed = distance / time * 3.6 # km/h

                if tracker_id in trackers_speed:
                    trackers_speed[tracker_id] = (trackers_speed[tracker_id] + speed) /2
                else:
                    trackers_speed[tracker_id] = speed

        
            if tracker_id in trackers_speed:
                speed_labels.append(f"{tracker_id}: {trackers_speed[tracker_id]:.0f} km/h")
            else:
                speed_labels.append(f"{tracker_id}: -- km/h")

        annotated_frame = frame.copy()
        annotated_frame = trace_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        annotated_frame = bounding_box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=speed_labels
        )

        # detect number plates
        for detection in detections:
            x1,y1,x2,y2 = np.array(detection[0]).astype(np.int32).tolist()
            roi = frame.copy()[y1:y2, x1:x2]

            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            res, mask = cv2.threshold(roi_gray, 100, 255, cv2.THRESH_BINARY)

            def mask_contours(mask, function):
                mask = mask.copy()
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours_to_remove = filter(lambda c: function(c), contours)
                contours_to_remove = list(contours_to_remove)
                return cv2.drawContours(mask, contours_to_remove, -1, (0, 0, 0), -1)

            mask = mask_contours(mask, lambda c: cv2.contourArea(c) < 500)
            mask = mask_contours(mask, lambda c: cv2.contourArea(c) > 2000)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


            for contour in contours:
                x,y,w,h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 2.5 < aspect_ratio < 4.5:
                    cv2.rectangle(annotated_frame, (x1+x, y1+y), (x1+x+w, y1+y+h), (0, 255, 0), 2)

        cv2.imshow("frame", annotated_frame)    

        plt.legend()
        plt.show()
        plt.pause(0.0001)

        if SAVE:
            sink.write_frame(annotated_frame)