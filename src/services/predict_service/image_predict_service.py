import os
import cv2
from model.base_model import BaseModel
from services.predict_service.predict import PredictService
from utils.get_file_path import get_file_path
from PIL import Image
import supervision as sv


class ImagePredictService(PredictService):
    CLASSES = {
        0: "Person"
    }

    def __init__(self, model: BaseModel, train_folder: str) -> None:
        super().__init__(model)
        self.train_folder = train_folder

    def predict(self, image_name: str) -> None:
        self.create_result_folder()

        image_file_path = get_file_path(
            f'images/{image_name}')
        img = cv2.imread(image_file_path)

        results = self.model.predict(image_file_path)

        detections = sv.Detections.from_yolov8(results[0])

        labels = [f'{self.CLASSES[class_id]} {confidence:0.2f}' for _, _, confidence, class_id, _ in detections]

        bounding_box_annotator = sv.BoxAnnotator(thickness=4)
        frame = bounding_box_annotator.annotate(scene=img, labels=labels, detections=detections, skip_label=True)

        image = Image.fromarray(frame[..., ::-1])
        image.save(os.path.join(self.result_folder, image_name))

    def create_result_folder(self) -> None:
        self.result_folder = os.path.join(
            os.getcwd(), 'results', 'images', self.train_folder)
        result_folder_exists = os.path.exists(self.result_folder)
        if result_folder_exists:
            return
        os.mkdir(self.result_folder)
