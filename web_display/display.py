from web_display.server import app
from dot_vision import ensemble_model
import random
from flask import Response, jsonify

@app.route('/video_stream')
def video_stream():    
    return Response(ensemble_model(is_stream_as_image = True), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/data")
def get_position():
    transformed_points = ensemble_model()
    print(f"transformed_points = {transformed_points}")
    position = {
        "x": random.randint(0, 1000),
        "y": random.randint(0, 1000),
        "initial_width": random.randint(0, 1000),
        "initial_height": random.randint(0, 1000),
    }
    return jsonify(position=position)
