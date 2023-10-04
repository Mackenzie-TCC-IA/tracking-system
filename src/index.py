import os
from model.yolo_model import YoloModel
from PIL import Image
from tap import Tap
import supervision as sv
import cv2


class YoloPeopleRecognitionArgs(Tap):
    use_pre_trained: bool = True
    model: str = 'yolov8n.pt'
    train_folder: str = 'train'
    dataset: str = 'data.yaml'
    image_name: str = ''
    epochs: int = 3
    video: str = ''


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

    is_video_mode = len(arguments.video) > 0

    model.evaluate()

    if not is_video_mode:
        results = model.predict(get_file_path(
            f'images/{arguments.image_name}'))

        for result in results:
            im_array = result.plot()
            im = Image.fromarray(im_array[..., ::-1])
            im.show()
            im.save(get_file_path(f'results/{arguments.image_name}'))
    else:
        video_path = get_file_path(f'videos/{arguments.video}')
        capture = cv2.VideoCapture(0)

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        capture.set(cv2.CAP_PROP_FPS, 30)

        while capture.isOpened():
            success, frame = capture.read()
            if success:
                results = model.predict(frame, use_nms=True)

                annotated_frame = results[0].plot()
                cv2.imshow("YOLOv8 Inference", annotated_frame)

                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break
            else:
                break

        capture.release()

cv2.destroyAllWindows()
