from abc import ABC, abstractmethod

from model.base_model import BaseModel


class PredictService(ABC):

    def __init__(self, model: BaseModel) -> None:
        self.model = model

    @abstractmethod
    def predict(self, *args, **kwargs) -> None:
        pass
