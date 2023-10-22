

from abc import ABC, abstractmethod
from typing import Any, TypedDict


class ModelProps(TypedDict):
    model_name: str
    dataset_path: str


class BaseModel(ABC):
    model: Any
    model_name: str
    dataset_path: str

    def __init__(self, model_props: ModelProps) -> None:
        self.model_name = model_props['model_name']
        self.dataset_path = model_props['dataset_path']

    @abstractmethod
    def train(self, epochs: int = 3, load: bool = False) -> None:
        pass

    @abstractmethod
    def evaluate(self) -> None:
        pass

    @abstractmethod
    def predict(self, args) -> Any:
        pass

    @abstractmethod
    def track(self, args) -> Any:
        pass
