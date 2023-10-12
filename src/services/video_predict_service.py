import cv2
from model.base_model import BaseModel
from services.predict import PredictService
from utils.get_file_path import get_file_path


class VideoPredictService(PredictService):
    def __init__(self, model: BaseModel) -> None:
        super().__init__(model)

    def predict(self, video_path: str) -> None:
        video_file_path = get_file_path(f'videos/{video_path}')
        capture = cv2.VideoCapture(video_file_path)

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        capture.set(cv2.CAP_PROP_FPS, 30)

        while capture.isOpened():
            success, frame = capture.read()
            if success:
                results = self.model.predict(
                    frame, use_nms=True)  # type: ignore

                annotated_frame = results[0].plot()
                cv2.imshow("YOLOv8 Inference", annotated_frame)

                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break
            else:
                break

        capture.release()
        cv2.destroyAllWindows()
