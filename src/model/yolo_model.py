import os
import re
from typing import Any
from ultralytics import YOLO
from model.base_model import BaseModel, ModelProps


class YoloModel(BaseModel):
    model: YOLO

    def __init__(self, model_props: ModelProps) -> None:
        super().__init__(model_props)

    def train(self, epochs: int = 3, load: bool = False) -> None:
        self.__set_runs_folder()

        if not load and self.__is_already_trained():
            self.model = YOLO(self.model_name)
            self.model.train(data=self.dataset_path, epochs=epochs)
            return

        self.model = YOLO('yolov8n.yaml').load(self.__get_best_weight_path())

    def __set_runs_folder(self) -> None:
        self.runs_folder = os.path.join(os.getcwd(), 'runs')

    def __is_already_trained(self) -> bool:
        return os.path.isdir(self.runs_folder)

    def __get_best_weight_path(self) -> str:
        detections_folder = os.path.join(self.runs_folder, 'detect')

        best_train = list(filter(self.__filter_train_folders,
                          os.listdir(detections_folder)))[-1]

        weights_path = os.path.join(
            detections_folder, best_train, 'weights', 'best.pt')

        return weights_path

    def __filter_train_folders(self, folder: str) -> bool:
        is_train_folder = re.search("^train[0-9]$", folder)
        return is_train_folder is not None

    def evaluate(self) -> Any:
        return self.model.val()

    def predict(self, args: Any) -> Any:
        return self.model.predict(args)
