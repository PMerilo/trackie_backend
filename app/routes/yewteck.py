from flask import Blueprint, Flask, request, jsonify
import numpy as np
from PIL import Image
import tempfile
import logging
import os
import requests
import urllib

from app.models.yewteck.load_model import load_action_model
import requests
import shutil


# Initialize logging
# logging.basicConfig(level=logging.INFO)
yewteck = Blueprint('yewteck', __name__, url_prefix='/yewteck')
model = load_action_model()

# Define the action labels
ACTIONS = [
    'sitting', 'using_laptop', 'hugging', 'sleeping', 'drinking',
    'clapping', 'dancing', 'cycling', 'calling', 'laughing',
    'eating', 'fighting', 'listening_to_music', 'running', 'texting'
]

def read_image(file_path):
    image = Image.open(file_path)
    image = image.resize((160, 160))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@yewteck.route('/detect-actions', methods=['GET'])
def detect_actions():
    src = request.args.get("src")
    file_path = os.path.join(tempfile.gettempdir(), 'predict.jpg')
    # r = requests.get('http://localhost:1984/api/frame.jpeg?src=phone')
    # if r.status_code == 200:
    #     with open(file_path, 'wb') as f:
    #         r.raw.decode_content = True
    #         shutil.copyfileobj(r.raw, f)   
    urllib.request.urlretrieve(f'{os.getenv("RTSP_SERVER_URL")}/api/frame.jpeg?src={src}', file_path)
    # file = request.files['file']
    # filename = tempfile.mktemp(suffix='.jpg')
    # file.save(filename)
    
    try:
        # Process the image and predict the action
        img_array = read_image(file_path)

        # Log the preprocessed image shape
        logging.info(f'Image shape: {img_array.shape}')

        # Save the preprocessed image for debugging
        np.save('debug_preprocessed.npy', img_array)

        predictions = model.predict(img_array)[0]
        # Log the raw predictions
        logging.info(f'Raw predictions: {predictions}')

        # Map predictions to probabilities
        actions_probabilities = {ACTIONS[i]: float(predictions[i]) for i in range(len(ACTIONS))}
        # Log the mapped probabilities
        logging.info(f'Predicted action probabilities: {actions_probabilities}')

        # Return the probabilities as a JSON response
        return jsonify(actions_probabilities)
    except Exception as e:
        logging.error(f'Error processing image: {str(e)}', exc_info=True)
        return jsonify({'error': 'Error processing image'}), 500
    finally:
        # Clean up: remove the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

@yewteck.route('/chat', methods=['POST'])
def chat():
    # Extract the JSON content from the incoming request
    incoming_data = request.json
    # Configure headers for the OpenAI API with your secret key
    headers = {
        'Authorization': f'Bearer {os.getenv("OPENAI_API_KEY")}',
        'Content-Type': 'application/json'
    }
    try:
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=incoming_data
        )
        response.raise_for_status()
        return jsonify(response.json()), response.status_code
    except requests.exceptions.HTTPError as err:
        # Log the error response from OpenAI API
        print(err.response.text)
        return jsonify({"error": "Error communicating with OpenAI API"}), 500
    
    # # Return the JSON response received from OpenAI API
    # return jsonify(response.json()), response.status_code