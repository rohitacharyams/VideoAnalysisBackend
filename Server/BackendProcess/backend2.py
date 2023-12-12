# Import necessary libraries
from flask import Flask, request, jsonify, send_file
import cv2
from flask_cors import CORS
import os
#from VideoProcessor import VideoProcessor

app = Flask(__name__)
CORS(app)

UPLOADS_FOLDER = "C://src//VideoLabellingBackend//Server//uploads"
app.config['UPLOADS_FOLDER'] = UPLOADS_FOLDER

keyframes_list = []

@app.route('/upload', methods=['POST'])
def upload_video():
    if request.method == 'POST':
        video_file = request.files['video']
        video_filename = video_file.filename
        video_path = f"../uploads/{video_filename}"

        # Save video file
        video_file.save(video_path)
        video_url = f"http://localhost:5000/uploads/{video_filename}"
        processed_video_url = f"C://src//VideoLabellingBackend//Server//uploads//{video_filename}"

        print(processed_video_url)

        return jsonify({'videoUrl': video_url})
    
@app.route('/uploads/<filename>')
def get_video(filename):
    # Serve the video file
    return send_file(os.path.join(app.config['UPLOADS_FOLDER'], filename))

@app.route('/get_frame', methods=['POST'])
def get_frame():
    if request.method == 'POST':
        video_filename = request.json['videoFilename']
        frame_index = request.json['frameIndex']
        video_path = f"../uploads/{video_filename}"

        # Extract and return the specified frame
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()
        cap.release()

        if success:
            # Encode frame as JPEG
            encoded_frame = cv2.imencode('.jpg', frame)[1].tobytes()
            return jsonify({'frame': encoded_frame.decode('utf-8')})
        else:
            return jsonify({'error': 'Invalid frame index'})

@app.route('/get_frame_info', methods=['POST'])
def get_frame_info():
    if request.method == 'POST':
        video_filename = request.json['videoFilename']
        video_path = f"../uploads/{video_filename}"

        # Get video frame rate
        cap = cv2.VideoCapture(video_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        return jsonify({'frameRate': frame_rate})

@app.route('/save_keyframes', methods=['POST'])
def save_keyframes():
    if request.method == 'POST':
        global keyframes_list
        keyframes_data = request.json['keyframes']

        # Save keyframes data to the global list
        keyframes_list = keyframes_data
        print(keyframes_list)
        # video_path = os.path.join(app.config['UPLOADS_FOLDER'], keyframes_data[0]['video_filename'])
        # processor = VideoProcessor(keyframes_list, video_path)
        # processor.process_video()

        return jsonify({'message': 'Keyframes saved successfully'})

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
