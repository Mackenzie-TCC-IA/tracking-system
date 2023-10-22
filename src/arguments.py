from tap import Tap


class YoloPeopleRecognitionArgs(Tap):
    use_pre_trained: bool = False
    model: str = 'yolov8n.pt'
    train_folder: str = 'train'
    dataset: str = 'data.yaml'
    epochs: int = 3
    mode: str = 'image'
    image_name: str = ''
    video_name: str = ''
    use_webcam = False


class CrowdHumanDownloaderArgs(Tap):
    download: bool = False
    extract: bool = False
