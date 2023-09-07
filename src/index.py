import os
from model.yolo_model import YoloModel
from PIL import Image
from tap import Tap


class YoloPeopleRecognitionArgs(Tap):
    use_pre_trained: bool = True
    model: str = 'yolov8n.pt'
    train_folder: str = 'train'
    dataset: str = 'data.yaml'
    image_name: str
    epochs: int = 3


def get_file_path(file_name: str) -> str:
    return os.path.join(os.getcwd(), file_name)


if __name__ == '__main__':
    arguments = YoloPeopleRecognitionArgs('yolo_face_recognition').parse_args()

    dataset_path = get_file_path(arguments.dataset)

    model = YoloModel({
        'model_name': arguments.model,
        'dataset_path': dataset_path
    })

    model.train(arguments.train_folder, arguments.epochs,
                arguments.use_pre_trained)

    results = model.predict(get_file_path(
        f'images/{arguments.image_name}'))

    model.evaluate()

    for result in results:
        im_array = result.plot()
        im = Image.fromarray(im_array[..., ::-1])
        im.show()
        im.save(get_file_path(f'results/{arguments.image_name}'))
