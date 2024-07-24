from web_display.server import app
from dot_vision import ensemble_model
import random
from flask import Response, jsonify, request


@app.route('/video_stream')
def video_stream():    
    return Response(ensemble_model(is_stream_as_image = True), mimetype='multipart/x-mixed-replace; boundary=frame')

"""
steps:
1. image height (h) and width (w)
2. screen height (h') and width (w')
3. divide (h') and (h) to get scale (p)
4. divide (w') and (w) to get scale (q)
5. use (p) and (q) to calculate (x) and (y) on the client side
6. where (x) = (p \cdot w) and (y) = (q \cdot h)
"""
@app.route("/data")
def get_position():
    # Fetch width and height from query parameters
    w_prime = request.args.get('width', default=1000, type=int)
    h_prime = request.args.get('height', default=1000, type=int)
    
    print(f"ensemble_model.image2d = {ensemble_model.image2d.shape}")
    
    h = ensemble_model.image2d.shape[0]
    w = ensemble_model.image2d.shape[1]
    h_scalar = h_prime / h
    w_scalar = w_prime / w
    
    transformed_points = ensemble_model()
    
    print(f"h_scalar = {h_scalar} w_scalar = {w_scalar} w = {w} h = {h} w_prime = {w_prime} h_prime = {h_prime}")
    print(f"transformed_points = {transformed_points}")
    
    if transformed_points is None:
        return jsonify(error="transformed_points is None"), 500
    
    try:
        points = [scale(point.flatten().astype(int).tolist(), w_scalar, h_scalar) for point in transformed_points]
        print(f"points = {points}")
    except Exception as e:
        print(f"error = {e}")
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