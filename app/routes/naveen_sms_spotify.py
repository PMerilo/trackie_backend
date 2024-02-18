from flask import Flask, request, jsonify, url_for
from flask_jwt_extended import JWTManager, create_access_token, set_access_cookies
import requests
from twilio.rest import Client
import os
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.environ.get('TWILIO_PHONE_NUMBER')
RECIPIENT_PHONE_NUMBER = os.environ.get('RECIPIENT_PHONE_NUMBER')

@app.route('/send_sms', methods=['POST'])
def send_sms():
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



if __name__ == '__main__':
    app.run(debug=True)