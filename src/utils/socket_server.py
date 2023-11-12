import asyncio
from websockets import serve


class WebSocketServer:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port

    async def start(self) -> None:
        async with serve(self.handle, self.host, self.port):
            print('Server connected')
            await asyncio.Future()

    async def handle(self, websocket) -> None:
        async for message in websocket:
            print(message)
