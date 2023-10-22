from ast import Tuple
import cv2
from model.base_model import BaseModel
from services.predict_service.predict import PredictService
from utils.get_file_path import get_file_path
from deep_sort_realtime.deepsort_tracker import DeepSort
from time import time
import supervision as sv


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

        bounding_box_annotator = sv.BoxAnnotator(
            thickness=4, text_thickness=2, text_scale=1)

        byte_tracker = sv.ByteTrack()

        previous_time = 0.0
        current_time = 0.0

        while capture.isOpened():
            success, frame = capture.read()

            if success:
                results = self.model.track(frame)[0]

                detections = sv.Detections.from_yolov8(results)
                detections = byte_tracker.update_with_detections(detections)

                labels = [
                    f'#{track_id} {self.CLASSES[class_id]} {confidence:0.2f} '
                    for _, _, confidence, class_id, track_id in detections
                ]

                frame = bounding_box_annotator.annotate(
                    scene=frame, detections=detections, labels=labels)

                current_time = time()
                fps = str(int(1 / (current_time - previous_time)))
                previous_time = current_time
                cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (100, 255, 0), 2)

                cv2.imshow("YOLOv8 Inference", frame)

                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break
            else:
                break

        capture.release()
        cv2.destroyAllWindows()
