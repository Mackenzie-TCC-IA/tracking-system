import os
import re
from typing import Any
from ultralytics import YOLO
from model.base_model import BaseModel, ModelProps


class YoloModel(BaseModel):
    model: YOLO

    def __init__(self, model_props: ModelProps) -> None:
        super().__init__(model_props)

    def train(self, train_model: str, epochs: int, pre_trained: bool = False) -> None:
        self.__set_runs_folder()

        if not pre_trained:
            self.model = YOLO(self.model_name)
            self.model.train(data=self.dataset_path, epochs=epochs)
            return

        self.model = YOLO(os.path.join(
            self.runs_folder, 'detect', train_model, 'weights', 'best.pt'))

    def __set_runs_folder(self) -> None:
        self.runs_folder = os.path.join(os.getcwd(), 'runs')

    def evaluate(self) -> Any:
        return self.model.val()

    def predict(self, args: Any, use_nms: bool = False) -> Any:
        return self.model.predict(args, agnostic_nms=use_nms)
