import asyncio
import json
import threading
import time
from typing import Any

import websockets
from arguments import YoloPeopleRecognitionArgs
from model.yolo_model import YoloModel
from services.predict_service.image_predict_service import ImagePredictService
from services.predict_service.stream_predict_service import StreamPredictService
from services.predict_service.video_predict_service import VideoPredictService
from utils.get_file_path import get_file_path


stream_predict_service: Any = None


async def handler(websocket: websockets.WebSocketServerProtocol):
    stream_predict_service.set_websocket(websocket)

    async for data in websocket:
        event = json.loads(data)

        if event['type'] == 'connect':
            await websocket.send(json.dumps(
                {'type': 'detections', 'data': stream_predict_service.current_detections}))
        elif event['type'] == 'update_name':
            stream_predict_service.change_name(
                event['data']['id'], event['data']['name'])


def server_callback():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    websocket_server = websockets.serve(handler, 'localhost', 8822)

    loop.run_until_complete(websocket_server)
    loop.run_forever()
    loop.close()


async def run_predict_service(predict_service: StreamPredictService, file_name: str):
    await predict_service.predict(file_name)


def predict_callback(predict_service: Any, file_name: str):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_predict_service(predict_service, file_name))
    loop.close()


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

    if arguments.mode == 'stream':
        stream_predict_service = predict_service['stream']

        server = threading.Thread(target=server_callback, daemon=True)
        server.start()
        time.sleep(2)

        predict = threading.Thread(target=predict_callback(
            stream_predict_service, file_name), daemon=True)

        predict.start()
        predict.join()
