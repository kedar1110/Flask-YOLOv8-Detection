from flask import Flask, render_template, request, Response, url_for
from werkzeug.utils import secure_filename
import os
import cv2
from ultralytics import YOLO
from detect_image import detect_image
from detect_video import detect_video

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DEFAULT_MODEL_PATH = 'yolov8n.pt'
CUSTOM_MODEL_PATH = 'runs/detect/train/weights/best.pt'

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/detect_image", methods=["POST"])
def handle_image():
    file = request.files.get('image')
    model_choice = request.form.get('model_choice', 'default')
    model_path = CUSTOM_MODEL_PATH if model_choice == 'custom' else DEFAULT_MODEL_PATH
    model = YOLO(model_path)

    if not file or file.filename == '':
        return "No file uploaded", 400

    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    result_path = detect_image(model, path, UPLOAD_FOLDER)
    return render_template('index.html', image_path=f'uploads/{os.path.basename(result_path)}')

@app.route("/detect_video", methods=["POST"])
def handle_video():
    file = request.files.get('video')
    model_choice = request.form.get('model_choice', 'default')
    model_path = CUSTOM_MODEL_PATH if model_choice == 'custom' else DEFAULT_MODEL_PATH
    model = YOLO(model_path)

    if not file or file.filename == '':
        return "No file uploaded", 400

    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    result_path = detect_video(model, path, UPLOAD_FOLDER)
    return render_template('index.html', video_path=f'uploads/{os.path.basename(result_path)}')

@app.route("/webcam")
def webcam():
    model_choice = request.args.get('model_choice', 'default')
    model_path = CUSTOM_MODEL_PATH if model_choice == 'custom' else DEFAULT_MODEL_PATH
    model = YOLO(model_path)
    return Response(gen_frames(model), mimetype='multipart/x-mixed-replace; boundary=frame')


def gen_frames(model):
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        results = model(frame)
        for r in results:
            annotated_frame = r.plot()
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

if __name__ == '__main__':
    app.run(debug=True, port=5000)

