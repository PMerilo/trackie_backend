from flask import Blueprint, Flask, request, jsonify, send_from_directory
import urllib
from werkzeug.utils import secure_filename
import os
import logging
import numpy as np
from PIL import Image
import tempfile
import io
from twilio.rest import Client
from app.models.naveen.load_model import load_emotion_model

# Configure logging
# logging.basicConfig(level=logging.INFO)
naveen = Blueprint('naveen', __name__, url_prefix='/naveen')


# Load your trained model
model = load_emotion_model()
# Specify the allowed extensions for uploaded files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @naveen.route('/', defaults={'path': ''})
# @naveen.route('/<path:path>')
# def serve(path):
#     if path and os.path.exists(naveen.static_folder + '/' + path):
#         return send_from_directory(naveen.static_folder, path)
#     return send_from_directory(naveen.static_folder, 'index.html')

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

@naveen.route('/detect-emotions', methods=['GET'])
def detect_emotions():
    src = request.args.get("src")
    # if 'file' not in request.files:
    #     return jsonify({'error': 'No file part in the request'}), 400

    # file = request.files['file']
    # if file.filename == '' or not allowed_file(file.filename):
    #     return jsonify({'error': 'No selected file or invalid file type'}), 400

    # filename = secure_filename(file.filename)
    file_path = os.path.join(tempfile.gettempdir(), 'predict.jpg')
    urllib.request.urlretrieve(f'{os.getenv("RTSP_SERVER_URL")}/api/frame.jpeg?src={src}', file_path)
    
    try:
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

@naveen.route('/send_sms', methods=['POST'])
def send_sms():
    TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID')
    TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN')
    TWILIO_PHONE_NUMBER = os.environ.get('TWILIO_PHONE_NUMBER')
    RECIPIENT_PHONE_NUMBER = os.environ.get('RECIPIENT_PHONE_NUMBER')

    data = request.json
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

    try:
        message = client.messages.create(
            body=data['message'],
            from_=TWILIO_PHONE_NUMBER,
            to=RECIPIENT_PHONE_NUMBER
        )
        return jsonify({'success': True, 'sid': message.sid}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    
# @app.route('/spotify/callback', methods=['GET'])
# def spotify_callback():
    
    # SPOTIFY_REDIRECT_URI = os.environ.get('SPOTIFY_REDIRECT_URI')
    # SPOTIFY_CLIENT_ID = os.environ.get('SPOTIFY_CLIENT_ID')
    # SPOTIFY_CLIENT_SECRET = os.environ.get('SPOTIFY_CLIENT_SECRET')
#     # Extract the authorization code from the callback URL
#     code = request.args.get('code')

#     # Exchange the authorization code for an access token
#     response = requests.post('https://accounts.spotify.com/api/token', data={
#         'grant_type': 'authorization_code',
#         'code': code,
#         'redirect_uri': os.environ.get('SPOTIFY_REDIRECT_URI'),
#         'client_id': os.environ.get('SPOTIFY_CLIENT_ID'),
#         'client_secret': os.environ.get('SPOTIFY_CLIENT_SECRET'),
#     })

#     if response.ok:
#         access_token = response.json()['access_token']
        
#         # Create a JWT token as a response
#         jwt_token = create_access_token(identity=access_token)
        
#         # Redirect to frontend
#         response = jsonify({'login': True})
#         set_access_cookies(response, jwt_token)
#         return response, 302, {'Location': url_for('frontend_route')}
#     else:
#         return jsonify({'error': 'Failed to retrieve access token from callback'}), 400