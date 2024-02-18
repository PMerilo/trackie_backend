from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import logging
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import tempfile
import io

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__, static_folder='build')
CORS(app)

# Update CORS settings as needed

# Load your trained model
MODEL_PATH = 'trained_model/model3.h5'  # Ensure this path is correct
model = load_model(MODEL_PATH)

# Specify the allowed extensions for uploaded files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

@app.route('/detect-emotions', methods=['POST'])
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

if __name__ == '__main__':
    app.run(debug=True)