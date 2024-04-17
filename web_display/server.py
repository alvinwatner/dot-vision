from flask import Flask, Response, render_template
from dot_vision_stream import gen_frames, app

@app.route('/video_stream')
def video_stream():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.get('/')
def index():
    return render_template("./index.html")


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3001, debug=True)
