from fastapi import FastAPI
import uvicorn
from fastapi.routing import APIRoute
from app.api.websocket_routes import router

app = FastAPI()

app.include_router(router)


def check_registered_routes():
    for route in app.routes:
        if isinstance(route, APIRoute):
            print(f"HTTP Route: {route.path}")
        else:
            print(f"WebSocket Route: {route.path}")


if __name__ == '__main__':
    check_registered_routes()
    uvicorn.run(app, host="127.0.0.1", port=5000)
