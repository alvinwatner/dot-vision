import threading
from new_dot_vision import args, ensemble_model
from web_display.api_provider import app


# needed for threading to work
def run_flask_app():
    app.run(host='0.0.0.0', port=3000, debug=False)


def run_ensemble_model():
    ensemble_model.stream_using_cv2()


def debug_tracking_with_cv2():
    process_web = threading.Thread(target=run_flask_app)
    process_cv2 = threading.Thread(target=run_ensemble_model)
    process_web.start()
    process_cv2.start()
    process_web.join()
    process_cv2.join()


debug = True

if __name__ == '__main__':
    if args.display == 'web':
        if debug:
            debug_tracking_with_cv2()
        else:
            run_flask_app()

    if args.display == 'cv2':
        ensemble_model.stream_using_cv2()
