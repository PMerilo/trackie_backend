from flask import Blueprint, request, jsonify
import os
import csv
from datetime import datetime
import subprocess
import sys
import signal
# from app.models.perry.predict import predict_bert, predict_svm
from app import transcriptions
TRANSCRIPTIONS_PATH = os.path.dirname(transcriptions.__file__)
#FLASK SECTION

process = None
model = ''

perry = Blueprint('perry', __name__, url_prefix='/perry')

@perry.route('/user/<string:userId>/get-transcripts', methods=['GET'])
def get_transcripts(userId):
    if userId != "debug":
        userId = f'user{userId}'
    from_date = request.args.get('from')
    to_date = request.args.get('to')
    
    files = transcriptions.get_transcript_files(userId)
    
    transcripts = []
    if not files:
        return jsonify(transcripts)
    

    try:
        datemin = datetime.min.strftime("%d-%m-%Y")
        today = datetime.now().strftime("%d-%m-%Y")

        
        if from_date and from_date.lower() == "today":
            from_date = today
        if to_date and to_date.lower() == "today":
            to_date = today
        
        from_date = datetime.strptime(from_date, "%d-%m-%Y") if from_date else datetime.strptime(datemin, "%d-%m-%Y")
        to_date = datetime.strptime(to_date, "%d-%m-%Y") if to_date else datetime.strptime(today, "%d-%m-%Y")
    except:
        return "Invalid date format. Ensure format is %d-%m-%Y", 400
    files = map(lambda x: x.split(".")[0], files)
    files = filter(lambda x: from_date <= datetime.strptime(x, "%d-%m-%Y") and to_date >= datetime.strptime(x, "%d-%m-%Y"), files)


    for f in files:
        transcripts += transcriptions.filereader(TRANSCRIPTIONS_PATH+"/"+userId+"/"+f+".csv")

    return jsonify(transcripts)


# @perry.route('/predict', methods=['POST'])
# def predict():
#     text = request.form['data']
#     result = {
#         'svm': predict_svm(text),
#         'bert': predict_bert(text)
#     }
#     return jsonify(result)

@perry.route('/state', methods=['GET'])
def get_state():
    global process, model
    return jsonify({ "state": process is not None or process.poll() is not None, "model": model })


@perry.route('/start', methods=['POST'])
def start_script():

    global process, model
    if process is not None and process.poll() is None:
        return 'Script is already running!', 400

    # Path to the child script
    transcribe_path = TRANSCRIPTIONS_PATH + "/transcribe.py"
    src = request.args.get("src")
    model = request.args.get('model')
    # print(transcribe_path)
    # print(src)
    # print(model)
    # Start the child script
    print("Starting the transcribe script...")
    process = subprocess.Popen([sys.executable, transcribe_path, src, model])
    return 'Child transcribe started.', 200

@perry.route('/stop', methods=['POST'])
def stop_script():
    global process
    if process is None or process.poll() is not None:
        return 'Script is not running.', 400

    # Stop the child script
    print("Stopping the transcribe script...")
    os.kill(process.pid, signal.SIGINT)  # Send KeyboardInterrupt to the child script
    process.wait()  # Wait for the child process to finish
    return 'Child transcribe stopped.', 200