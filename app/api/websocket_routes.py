import json

from fastapi import WebSocket, status, WebSocketException, APIRouter
from app.models.request_model import FrameRequest
from app.config import logger
from app.services.homographic_service import HomographicService
from app.models.request_model import Coordinates

router = APIRouter()


async def proces_websocket_request(websocket: WebSocket, service: callable):
    await websocket.accept()
    while True:
        data = await websocket.receive_json()
        try:
            validated_data = FrameRequest(**data)
        except Exception as e:
            logger.error(f"error happened: {e}")
            raise WebSocketException(code=status.WS_1003_UNSUPPORTED_DATA, reason=f"Error occurred: {e}")
        result = service(validated_data.frame)
        await websocket.send_json({"result": result})


# TODO: get user data, find the coordinates from that user, use it for homographic transformation
@router.websocket("/ws/homographic")
async def homographic_websocket(websocket: WebSocket):
    with open("/home/kaorikizuna/Dot Vision/dot-vision/app/coordinates/example_coordinates.json", "r") as f:
        example_data = json.load(f)

    validated_coordinates_data = Coordinates(**example_data)

    homographic_service = HomographicService(coordinates=validated_coordinates_data)
    await proces_websocket_request(websocket, homographic_service.execute)


@router.websocket("/ws")
async def hello(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("Hello World")
