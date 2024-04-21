from web_display.server import app
from dot_vision import ensemble_model
from flask import Response

@app.route('/video_stream')
def video_stream():
    return Response(ensemble_model(is_stream = True), mimetype='multipart/x-mixed-replace; boundary=frame')


