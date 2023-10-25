from typing import Literal
from tap import Tap


class YoloPeopleRecognitionArgs(Tap):
    use_pre_trained: bool = False
    model: str = 'yolov8n.pt'
    train_folder: str = 'train'
    dataset: str = 'data-crowd-humans.yaml'
    epochs: int = 3
    mode: Literal['image', 'video', 'stream'] = 'image'
    image_name: str = ''
    video_name: str = ''


class CrowdHumanDownloaderArgs(Tap):
    download: bool = False
    extract: bool = False
