from arguments import YoloPeopleRecognitionArgs
from model.yolo_model import YoloModel
from services.image_predict_service import ImagePredictService
from services.video_predict_service import VideoPredictService
from utils.get_file_path import get_file_path


if __name__ == '__main__':
    arguments = YoloPeopleRecognitionArgs('yolo_face_recognition').parse_args()

    dataset_path = get_file_path(arguments.dataset)

    model = YoloModel({
        'model_name': arguments.model,
        'dataset_path': dataset_path
    })

    model.train(arguments.train_folder, arguments.epochs,
                arguments.use_pre_trained)

    predict_service = {
        'image': ImagePredictService(model),
        'video': VideoPredictService(model)
    }

    model.evaluate()

    file_name = arguments.image_name if arguments.mode == 'image' else arguments.video_name

    predict_service[arguments.mode].predict(file_name)
