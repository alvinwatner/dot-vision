from web_display.server import app
from dot_vision_stream import gen_frames
from flask import Response

@app.route('/video_stream')
def video_stream():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


