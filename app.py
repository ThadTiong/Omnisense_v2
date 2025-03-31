import cv2
from flask import Flask, render_template, Response, jsonify, request, abort
from importlib import import_module
from camera import Camera, list_cameras

app = Flask(__name__)

# Initialize the default camera using DirectShow backend
current_camera = 0
camera = Camera()
camera.connect(0, cv2.CAP_ANY)

def load_module(mode):
    module_name = mode
    processing_module = import_module(module_name)
    processing_function = getattr(processing_module, f"process_{mode}")
    
    return processing_module, processing_function

def generate_frames(mode):
    processing_module, processing_function = load_module(mode)

    while True:
        frame = camera.get_frame()
        frame = processing_function(frame)
            
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed/<mode>')
def video_feed(mode):
    return Response(generate_frames(mode), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera_list')
def camera_list():
    camera_list = list_cameras()
    return jsonify(camera_list)

@app.route('/switch_camera/<int:camera_id>')
def switch_camera(camera_id):
    global current_camera
    
    print(f"Switching to camera {camera_id}")
    
    # Define backend preferences in the order of preference
    if camera.switch_camera(camera_id):
        current_camera = camera_id
    else:
        current_camera = 0
        camera.connect(current_camera, cv2.CAP_ANY)
        return jsonify({'error': f"Failed to open camera {camera_id}. Switched to default camera."}), 409
    
    return jsonify({'message': f'Switched to camera {camera_id}.'})

if __name__ == '__main__':
    app.run(debug=True)
