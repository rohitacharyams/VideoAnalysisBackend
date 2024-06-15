# Import necessary libraries
from flask import Flask, request, jsonify, send_file, send_from_directory, Response
import cv2
from flask_cors import CORS
import os
from flask_bcrypt import Bcrypt
from flask_pymongo import PyMongo
from VideoProcessor import VideoProcessor
from pymongo.mongo_client import MongoClient
from pymongo import ASCENDING
import subprocess
import urllib.parse
import uuid
import datetime
import gridfs
from bson import ObjectId
import io
from azure.storage.blob import BlobServiceClient
import random
from io import BytesIO
import requests
import numpy as np
import tempfile
import re, shutil, json
from azure.data.tables import TableServiceClient, UpdateMode


app = Flask(__name__)
CORS(app)
bcrypt = Bcrypt(app)

from pymongo import MongoClient
import urllib.parse


uri = "get_from_whatsapp"

# Create a new client and connect to the server
client = MongoClient(uri)
db = client['dance_database']
video_collection = db['videos']
fs = gridfs.GridFS(db)

UPLOADS_FOLDER = "/Users/rohitacharya/danceAI/VideoAnalysisBackend-main/Server/uploads"
app.config['UPLOADS_FOLDER'] = UPLOADS_FOLDER
current_video_path = None

# Azure Storage connection string
connection_string = "get_from_whatsapp_source"
blob_service_client_source = BlobServiceClient.from_connection_string(connection_string)
container_name = "aistdancevideos"

connection_string_table = "get_from_whatsapp_destination" # Both of them are different
blob_service_client_dest = BlobServiceClient.from_connection_string(connection_string_table)


source_container_name = "aistdancevideos"
destination_container_name = "labellingdone"

# Azure Table Storage setup
table_service_client = TableServiceClient.from_connection_string(connection_string_table)
table_name = "LabeledVideos"
table_client = table_service_client.create_table_if_not_exists(table_name=table_name)



keyframes_list = []
temp_dir = tempfile.mkdtemp()

# Functions

def query_all_entities():
    entities = table_client.list_entities()
    for entity in entities:
        video_id = entity['RowKey']
        video_filename = entity['VideoFilename']
        keyframes_json = entity['Keyframes']
        keyframes = json.loads(keyframes_json)
        print(f"Video ID: {video_id}")
        print(f"Video Filename: {video_filename}")
        print(f"Keyframes: {keyframes}")
        print("-------------------------------")


def save_keyframes_to_table(video_id, video_filename, keyframes):
    keyframes_json = json.dumps(keyframes)
    entity = {
        'PartitionKey': 'LabeledVideos',
        'RowKey': video_id,
        'VideoFilename': video_filename,
        'Keyframes': keyframes_json
    }
    table_client.upsert_entity(entity=entity, mode=UpdateMode.MERGE)

    # Uncomment below if you want to see all entries of Azure Table Storage we are using 
    # query_all_entities()

def check_video_labeled(video_filename):
    query_filter = f"PartitionKey eq 'LabeledVideos' and VideoFilename eq '{video_filename}'"
    entities = table_client.query_entities(query_filter=query_filter)
    return any(entities)

def move_video_to_new_container(video_filename):
    source_blob_client = blob_service_client_source.get_blob_client(container=source_container_name, blob=video_filename)
    destination_blob_client = blob_service_client_dest.get_blob_client(container=destination_container_name, blob=video_filename)

    source_blob_data = source_blob_client.download_blob().readall()
    destination_blob_client.upload_blob(source_blob_data, overwrite=True)
    source_blob_client.delete_blob()


def serialize_video(video):
    video['_id'] = str(video['_id'])
    return video


def run_mmpose(video_path):
    path_to_mmpose = "/Users/rohitacharya/mmpose"
    python_path = "/opt/homebrew/opt/python@3.10/bin/python3.10"
    mmpose_command = [
        python_path, f"{path_to_mmpose}/demo/topdown_demo_with_mmdet.py",
        f"{path_to_mmpose}/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py",
        "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth",
        f"{path_to_mmpose}/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py",
        "/Users/rohitacharya/Downloads/model_full_body.pth",
        "--input", video_path,
        "--show"
    ]

    print("Heywouhekwuh")

    print("command is :", mmpose_command)

    subprocess.run(mmpose_command)


def fetch_keypoints_from_mongodb(video_id, track_id):
    video = video_collection.find_one({'video_id': video_id})
    if video:
        frames = video['frames']
        keypoints = [frames[frame]['tracks'][track_id]['keypoints'] for frame in frames if track_id in frames[frame]['tracks']]
        return keypoints
    return None


# Routes 
@app.route('/api/videos', methods=['GET'])
def get_videos():
    videos = list(video_collection.find())
    video_list = []
    for video in videos:
        video_list.append({
            'video_id': video['video_id'],
            'title': video['title'],
            'upload_date': video['upload_date'],
            'uploaded_by': video['uploaded_by'],
        })
    print(video_list)
    return jsonify(video_list)

@app.route('/api/bbox_info/<video_id>', methods=['GET'])
def get_bbox_info(video_id):
    video = video_collection.find_one({'video_id': video_id})
    if video and 'bbox_info' in video:
        return jsonify(video['bbox_info'])
    return jsonify({'error': 'Bounding box information not found'}), 404

@app.route('/api/detect_dancer/<video_id>', methods=['POST'])
def detect_dancer(video_id):
    data = request.json
    x, y = data['x'], data['y']
    video = video_collection.find_one({'video_id': video_id})
    if video:
        for frame in video['frames'].values():
            for track_id, track_data in frame['tracks'].items():
                bbox = track_data['bbox']
                if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                    keypoints = track_data['keypoints']
                    return jsonify({'dancerId': track_id, 'keypoints': keypoints})
        return jsonify({'error': 'Dancer not found'}), 404
    else:
        return jsonify({'error': 'Video not found'}), 404

@app.route('/api/video/<video_id>', methods=['GET'])
def get_reel(video_id):
    print("Say HIisd baby")
    video = video_collection.find_one({'video_id': video_id})
    if not video:
        return jsonify({'error': 'Video not found'}), 404
    
    print("Video id is ", video_id)

    video_path = os.path.join(app.config['UPLOADS_FOLDER'], video['title'])

    # video_path = video_path + '.mp4'

    print("Video path is ", video_path)

    if not os.path.exists(video_path):
        return jsonify({'error': 'Video file not found'}), 404
    
    print("Video path is sent successfully")

    return send_file(video_path)

def clear_temp_files():
    global current_video_filename
    for filename in os.listdir(f"{UPLOADS_FOLDER}"):
        file_path = os.path.join(f"{UPLOADS_FOLDER}", filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
    current_video_filename = None

@app.route('/api/fetch_video', methods=['POST'])
def fetch_video():
    global current_video_filename
    data = request.json
    video_url = data.get('url')
    if not video_url:
        return jsonify({"error": "No video URL provided"}), 400

    # Clear previous video
    clear_temp_files()

    response = requests.get(video_url, stream=True)
    video_filename = os.path.basename(video_url)
    video_path = os.path.join(f"{UPLOADS_FOLDER}", video_filename)
    with open(video_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    current_video_filename = video_filename
    video_url = f"http://localhost:51040/uploads/{video_filename}"
    return jsonify({"message": "Video fetched", "videoUrl": video_url})

@app.route('/api/get_video', methods=['GET'])
def get_video_from_local(filename):
    return send_from_directory(f"{UPLOADS_FOLDER}", filename)

@app.route('/api/clear_temp', methods=['POST'])
def clear_temp():
    clear_temp_files()
    return jsonify({"message": "Temporary files cleared"})


@app.route('/api/composite_thumbnail/<video_id>', methods=['GET'])
def get_composite_image(video_id):
    print(video_id)
    video = video_collection.find_one({'video_id': video_id})
    # print("Video:", video)
    if video and 'composite_image_path' in video:
        print("Composite image path is :",video['composite_image_path'])
        return send_file(video['composite_image_path'])
    return jsonify({'error': 'Composite image not found'}), 404

@app.route('/api/keypoints/<video_id>/<track_id>', methods=['GET'])
def get_keypoints(video_id, track_id):
    keypoints = fetch_keypoints_from_mongodb(video_id, track_id)
    if keypoints:
        print("Keypoints: ", (keypoints))
        return jsonify(keypoints)
    return jsonify({'error': 'Keypoints not found'}), 404


@app.route('/api/upload', methods=['POST'])
def upload_reels():
    print("Say HIisd baby")
    
    video_file = request.files['video']
    video_filename = video_file.filename
    video_id = str(uuid.uuid4())
    title = video_filename.split('.')[0]
    upload_date = datetime.datetime.now().strftime("%Y-%m-%d")
    uploaded_by = "admin"
    video_path = f"{UPLOADS_FOLDER}/{video_filename}"

    print("gh")
    video_data = {
        'video_id': video_id,
        'title': title,
        'upload_date': upload_date,
        'uploaded_by': uploaded_by,
    }
    # video_collection.insert_one(video_data)

    print("ghqw")

    run_mmpose(video_path)

    return jsonify({'message': 'Video uploaded and processed successfully'}), 201

@app.route('/upload', methods=['POST'])
def upload_video():
    if request.method == 'POST':
        video_file = request.files['video']
        video_filename = video_file.filename
        video_path = f"{UPLOADS_FOLDER}/{video_filename}"

        # Save video file
        video_file.save(video_path)
        video_url = f"http://localhost:51040/uploads/{video_filename}"
        processed_video_url = f"C://src//VideoLabellingBackend//Server//uploads//{video_filename}"

        print("Processed video url", processed_video_url)

        return jsonify({'videoUrl': video_url})
    
@app.route('/uploads/<filename>')
def get_video(filename):
    # Serve the video file
    print("Why this is getting called", filename)
    return send_file(os.path.join(app.config['UPLOADS_FOLDER'], filename))

@app.route('/api/videosFromStorage')
def get_videos_from_blob():
    container_client = blob_service_client.get_container_client(container_name)
    blob_list = [blob.name for blob in container_client.list_blobs()]
    if not blob_list:
        return jsonify({"error": "No videos found"}), 404
    random_video = random.choice(blob_list)
    video_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{random_video}"
    print("Video url is :", video_url)
    return jsonify({"url": video_url, "videoFilename": random_video})

@app.route('/get_frame', methods=['POST'])
def get_frame():
    if request.method == 'POST':
        video_filename = request.json['videoFilename']
        frame_index = request.json['frameIndex']
        video_path = f"{UPLOADS_FOLDER}/{video_filename}"

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
        video_path = f"{UPLOADS_FOLDER}/{video_filename}"

        # Get video frame rate
        cap = cv2.VideoCapture(video_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        return jsonify({'frameRate': frame_rate})

def generate_guid():
    return str(uuid.uuid4())

@app.route('/save_keyframes', methods=['POST'])
def save_keyframes():
    if request.method == 'POST':
        try:
            global keyframes_list
            keyframes_data = request.json.get('keyframes', [])
            video_filename = request.json.get('video_filename', None)
            video_id = generate_guid()

            print("The values of keyframes_data, video_filename, video_id", keyframes_data, video_filename, video_id)

            if not keyframes_data or not video_filename or not video_id:
                return jsonify({'error': 'Missing keyframes data, video filename, or video ID'}), 400

            # Save keyframes data to Table Storage
            save_keyframes_to_table(video_id, video_filename, keyframes_data)

            # Move video to new container
            move_video_to_new_container(video_filename)

            return jsonify({'message': 'Keyframes saved and video moved successfully'})
        except Exception as e:
            print(f"An error occurred: {e}")
            return jsonify({"error": str(e)}), 500
    


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=51040, debug=True)
