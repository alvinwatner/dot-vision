from web_display.flask_app import app
from flask import Response, jsonify, request
from new_dot_vision import ensemble_model
from auxiliary.image_handler import ImageHandler
import logging

logging.basicConfig(level=logging.DEBUG)


@app.route('/video_stream')
def video_stream():
    return Response(ensemble_model.stream_as_image(), mimetype='multipart/x-mixed-replace; boundary=frame')


"""
steps:
1. image height (h) and width (w)
2. screen height (h') and width (w')
3. divide (h') and (h) to get scale (p)
4. divide (w') and (w) to get scale (q)
5. use (p) and (q) to calculate (x) and (y) on the client side
6. where (x) = (p \cdot w) and (y) = (q \cdot h)
"""


def perform_linear_transform(image_handler: ImageHandler, height_prime, width_prime):
    height_origin = image_handler.image2d.height
    width_origin = image_handler.image2d.width
    height_scalar = height_prime / height_origin
    width_scalar = width_prime / width_origin

    return height_scalar, width_scalar


@app.get("/data")
def get_position():
    # Fetch width and height from query parameters
    w_prime = request.args.get('width', default=1000, type=int)
    h_prime = request.args.get('height', default=1000, type=int)

    h_scalar, w_scalar = perform_linear_transform(ensemble_model.image_handler, height_prime=h_prime,
                                                  width_prime=w_prime)

    transformed_points = ensemble_model.generate_raw_outputs()

    logging.debug(transformed_points)
    try:
        points = [scale(point.flatten().astype(int).tolist(), w_scalar, h_scalar) for point in transformed_points]
        logging.debug(f"points = {points}")
    except Exception as e:
        logging.debug(f"error = {e}")
        return jsonify(error=str(e)), 500

    position = {
        "points": points,
    }

    return jsonify(position=position)


def scale(point, w_scalar, h_scalar):
    x = point[0]
    y = point[1]
    scaled_point = [int(x * w_scalar), int(y * h_scalar)]
    return scaled_point
