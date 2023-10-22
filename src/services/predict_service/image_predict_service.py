import os
from model.base_model import BaseModel
from services.predict_service.predict import PredictService
from utils.get_file_path import get_file_path
from PIL import Image


class ImagePredictService(PredictService):
    def __init__(self, model: BaseModel, train_folder: str) -> None:
        super().__init__(model)
        self.train_folder = train_folder

    def predict(self, image_name: str) -> None:
        self.create_result_folder()

        image_file_path = get_file_path(
            f'images/{image_name}')

        results = self.model.predict(image_file_path)

        for result in results:
            im_array = result.plot()
            image = Image.fromarray(im_array[..., ::-1])
            image.save(os.path.join(self.result_folder, image_name))

    def create_result_folder(self) -> None:
        self.result_folder = os.path.join(
            os.getcwd(), 'results', 'images', self.train_folder)
        result_folder_exists = os.path.exists(self.result_folder)
        if result_folder_exists:
            return
        os.mkdir(self.result_folder)
