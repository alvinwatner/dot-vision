from web_display.server import app
from dot_vision_stream import ensemble_model
from flask import Response

@app.route('/video_stream')
def video_stream():
    return Response(ensemble_model.run(is_stream = True), mimetype='multipart/x-mixed-replace; boundary=frame')


