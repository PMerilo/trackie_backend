import os
import csv
TRANSCRIPTIONS_PATH = os.path.dirname(__file__)

def filereader(path):
    list = []
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            list.append({ "timestamp" : row[0] , "transcript" : row[1], "predict" : row[2]})
    return list

def get_transcript_files(userId):
    folders = [ f for f in os.listdir(TRANSCRIPTIONS_PATH) if os.path.isdir(os.path.join(TRANSCRIPTIONS_PATH, f)) ] # Get only dir in transcript folder
    folders.remove("__pycache__")

    if userId is None:
        return folders
    
    if userId in folders:
        return os.listdir(TRANSCRIPTIONS_PATH+"/"+userId)
    else: 
        return []