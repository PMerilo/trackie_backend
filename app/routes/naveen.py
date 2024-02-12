from flask import Blueprint, Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import logging
import numpy as np
from PIL import Image
import tempfile
import io
from app.models.naveen.load_model import load_emotion_model

# Configure logging
logging.basicConfig(level=logging.INFO)
naveen = Blueprint('naveen', __name__, url_prefix='/naveen')


# Load your trained model
model = load_emotion_model()
# Specify the allowed extensions for uploaded files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@naveen.route('/', defaults={'path': ''})
@naveen.route('/<path:path>')
def serve(path):
    if path and os.path.exists(naveen.static_folder + '/' + path):
        return send_from_directory(naveen.static_folder, path)
    return send_from_directory(naveen.static_folder, 'index.html')

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

@naveen.route('/detect-emotions', methods=['POST'])
def detect_emotions():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'No selected file or invalid file type'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(tempfile.gettempdir(), filename)
    file.save(file_path)
    
    try:
        # img = Image.open(file_path).convert('L')
        # img = img.resize((48, 48))
        # img_array = np.stack((img,) * 3, axis=-1)
        # img_array = img_array / 255.0
        # img_array = np.expand_dims(img_array, axis=0)

        img = Image.open(file_path).convert('L')  # Convert to grayscale
        img = img.resize((48, 48))
        img_array = np.array(img) # Single channel grayscale 
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
        img_array = np.expand_dims(img_array, axis=0)   # Add batch dimension

        predictions = model.predict(img_array)[0]
        emotion_probabilities = {EMOTIONS[i]: float(predictions[i]) for i in range(len(EMOTIONS))}

        print(emotion_probabilities)
        return jsonify(emotion_probabilities)
    except Exception as e:
        logging.error(f'Error processing image: {str(e)}', exc_info=True)
        return jsonify({'error': 'Error processing image'}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)