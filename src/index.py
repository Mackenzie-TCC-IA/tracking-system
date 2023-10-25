from arguments import YoloPeopleRecognitionArgs
from model.yolo_model import YoloModel
from services.predict_service.image_predict_service import ImagePredictService
from services.predict_service.stream_predict_service import StreamPredictService
from services.predict_service.video_predict_service import VideoPredictService
from utils.get_file_path import get_file_path


if __name__ == '__main__':
    arguments = YoloPeopleRecognitionArgs('yolo_face_recognition').parse_args()

    model = YoloModel({
        'model_name': arguments.model,
        'dataset_path': get_file_path(arguments.dataset)
    })

    model.train(arguments.train_folder, arguments.epochs,
                arguments.use_pre_trained)

    predict_service = {
        'image': ImagePredictService(model, arguments.train_folder),
        'video': VideoPredictService(model),
        'stream': StreamPredictService(model)
    }

    if not arguments.use_pre_trained:
        model.evaluate()

    file_name = arguments.image_name if arguments.mode == 'image' else arguments.video_name

    predict_service[arguments.mode].predict(file_name)
