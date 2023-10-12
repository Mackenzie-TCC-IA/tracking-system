from model.base_model import BaseModel
from services.predict import PredictService
from utils.get_file_path import get_file_path
from PIL import Image


class ImagePredictService(PredictService):

    def __init__(self, model: BaseModel) -> None:
        super().__init__(model)

    def predict(self, image_name: str) -> None:
        image_file_path = get_file_path(
            f'images/{image_name}')

        results = self.model.predict(image_file_path)

        for result in results:
            im_array = result.plot()
            im = Image.fromarray(im_array[..., ::-1])
            im.show()
            im.save(get_file_path(f'results/{image_name}'))
