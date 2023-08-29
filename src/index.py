from typing import TypedDict
from model.yolo_model import YoloModel
from types import SimpleNamespace
from PIL import Image
import argparse


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser('yolo_face_recognition')
    argument_parser.add_argument(
        '--use_pre_trained', type=bool, help='Set model to use a pre trained if is not trained')

    arguments = argument_parser.parse_args()

    yolo_v8_model_file = 'yolov8n.pt'
    dataset_path = 'coco128.yaml'

    model = YoloModel({
        'model_name': yolo_v8_model_file,
        'dataset_path': dataset_path
    })

    model.train(4, arguments.use_pre_trained)

    results = model.predict('https://ultralytics.com/images/bus.jpg')

    for result in results:
        im_array = result.plot()
        im = Image.fromarray(im_array[..., ::-1])
        im.show()
        im.save('results/results.jpg')
