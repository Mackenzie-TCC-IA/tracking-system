from ast import Tuple
from typing import Any
import cv2
import numpy as np
from model.base_model import BaseModel
from services.predict_service.predict import PredictService
from utils.get_file_path import get_file_path
from deep_sort_realtime.deepsort_tracker import DeepSort
from time import time


class StreamPredictService(PredictService):
    CLASSES = {
        0: 'Person'
    }

    def __init__(self, model: BaseModel) -> None:
        super().__init__(model)

    def predict(self, video_name: str) -> None:
        is_webcam = len(video_name) == 0

        capture = cv2.VideoCapture(
            0 if is_webcam else get_file_path(f'videos/{video_name}'))

        if is_webcam:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
            capture.set(cv2.CAP_PROP_FPS, 30)

        tracker = DeepSort(max_age=5, nms_max_overlap=0.4)

        prev_frame_time = 0.0
        new_frame_time = 0.0

        while capture.isOpened():
            success, frame = capture.read()
            if success:
                results = self.model.predict(frame, use_nms=True)

                for result in results:
                    detections = []

                    for data in result.boxes.data.tolist():
                        x1, y1, x2, y2 = data[:4]
                        width, heigth = x2 - x1, y2 - y1
                        detections.append(
                            (list((int(x1), int(y1), int(width), int(heigth))), data[4], self.CLASSES[data[5]]))

                    tracks = tracker.update_tracks(detections, frame=frame)

                    for track in tracks:
                        if not track.is_confirmed():
                            continue
                        bbox = track.to_ltrb()

                        cv2.rectangle(
                            frame,
                            (int(bbox[0]), int(bbox[1])),
                            (int(bbox[2]), int(bbox[3])),
                            color=(0, 0, 255),
                            thickness=4,
                        )
                        cv2.putText(
                            frame,
                            f"#{track.track_id} {track.det_class}",
                            (int(bbox[0]), int(bbox[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )

                new_frame_time = time()

                fps = str(int(1 / (new_frame_time - prev_frame_time)))
                prev_frame_time = new_frame_time

                cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX,
                            2, (100, 255, 0), 2)

                cv2.imshow("YOLOv8 Inference", frame)

                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break
            else:
                break

        capture.release()
        cv2.destroyAllWindows()
