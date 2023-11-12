import json
from typing import Any
import cv2
from websockets import WebSocketServerProtocol
from model.base_model import BaseModel
from services.predict_service.predict import PredictService
from utils.get_file_path import get_file_path
from time import time
import supervision as sv


class StreamPredictService(PredictService):
    CLASSES = {
        0: 'Person'
    }

    current_detections: dict[str, Any] = {}

    websocket: WebSocketServerProtocol | None = None

    def __init__(self, model: BaseModel) -> None:
        super().__init__(model)

    async def predict(self, video_name: str) -> None:
        is_webcam = len(video_name) == 0

        capture = cv2.VideoCapture(
            0 if is_webcam else get_file_path(f'videos/{video_name}'))

        if is_webcam:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
            capture.set(cv2.CAP_PROP_FPS, 30)

        bounding_box_annotator = sv.BoxAnnotator(
            thickness=4, text_thickness=2, text_scale=0.7)

        byte_tracker = sv.ByteTrack()

        previous_time = 0.0
        current_time = 0.0

        while capture.isOpened():
            success, frame = capture.read()

            if success:
                results = self.model.track(frame)[0]

                detections = sv.Detections.from_yolov8(results)
                detections = byte_tracker.update_with_detections(detections)

                parsed_detections = dict(
                    [(str(track_id), 'Unknown') for track_id in detections.tracker_id])

                if len(self.current_detections) == 0:
                    self.current_detections = parsed_detections
                    if self.websocket:
                        await self.websocket.send(json.dumps({'type': 'detections',
                                                              'data': self.current_detections}))
                else:
                    new_current_detections_ids = set(parsed_detections.keys())
                    current_detection_ids = set(
                        self.current_detections.keys())

                    new_ids = new_current_detections_ids - current_detection_ids
                    removed_ids = current_detection_ids - new_current_detections_ids

                    for new_id in new_ids:
                        self.current_detections[new_id] = 'Unknown'

                    for remove_id in removed_ids:
                        self.current_detections.pop(remove_id)

                    if self.websocket:
                        if len(new_ids) > 0:
                            await self.websocket.send(json.dumps({
                                'type': 'new_detections',
                                'data': list(new_ids)
                            }))

                        if len(removed_ids) > 0:
                            await self.websocket.send(json.dumps({
                                'type': 'removed_detections',
                                'data': list(removed_ids)
                            }))

                labels = [
                    f'#{track_id} {self.current_detections[str(track_id)]} '
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

    def set_websocket(self, websocket: WebSocketServerProtocol) -> None:
        self.websocket = websocket

    def change_name(self, id: str, new_name: str) -> None:
        self.current_detections[id] = new_name
