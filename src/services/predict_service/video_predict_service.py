import numpy as np
from tqdm import tqdm
from model.base_model import BaseModel
from services.predict_service.predict import PredictService
from utils.get_file_path import get_file_path
import supervision as sv


class VideoPredictService(PredictService):
    CLASSES = {
        0: 'Person'
    }

    def __init__(self, model: BaseModel) -> None:
        super().__init__(model)

    def predict(self, video_name: str) -> None:
        video_full_path = get_file_path(f'videos/{video_name}')

        video_info = sv.VideoInfo.from_video_path(video_full_path)
        video_frame_generator = sv.get_video_frames_generator(video_full_path)

        polygon = np.array([
            [0, 0],
            [video_info.width, 0],
            [video_info.width, video_info.height],
            [0, video_info.height]
        ])

        zone = sv.PolygonZone(
            polygon=polygon, frame_resolution_wh=video_info.resolution_wh)

        zone_annotator = sv.PolygonZoneAnnotator(
            zone=zone, color=sv.Color.red(), thickness=2, text_thickness=5, text_scale=2)

        bounding_box_annotator = sv.BoxAnnotator(
            thickness=4, text_thickness=2, text_scale=1)

        byte_tracker = sv.ByteTrack()

        video_result_path = get_file_path(f'results/videos/{video_name}')

        with sv.VideoSink(target_path=video_result_path, video_info=video_info) as sink:
            for frame in tqdm(video_frame_generator, total=video_info.total_frames):
                results = self.model.predict(frame)[0]

                detections = sv.Detections.from_yolov8(results)
                detections = byte_tracker.update_with_detections(detections)

                labels = [
                    f'#{track_id}'
                    for _, _, confidence, class_id, track_id in detections
                ]

                frame = bounding_box_annotator.annotate(
                    scene=frame, detections=detections, labels=labels)

                sink.write_frame(frame=frame)
