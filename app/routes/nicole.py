import subprocess
import sys
from flask import Blueprint, request
import pandas as pd
import csv
import json
import numpy as np
import os
from app.models.nicole.load_model import getHobbiesFromIds, getUserHobbies, load_recommender_model
from app import models

MODEL_PATH = os.path.dirname(models.__file__)+"/nicole"

nicole = Blueprint('nicole', __name__, url_prefix='/nicole')

@nicole.route("/add-hobbies", methods=['POST'])
def add_hobbies():
    try:
        user = request.get_json()
        user_id = user['id']
        print(f"Add hobbies for user: {user_id}")
        rows = [ [user_id+5000, hobbyId, 3] for hobbyId in user['hobbies']]

        with open(MODEL_PATH + "/Data/User_Preferrences.csv", 'a', newline='' ) as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)
        subprocess.Popen([sys.executable, MODEL_PATH + "/train_model.py"])
        return "Success"
    except:
        return "Error", 500


@nicole.route("/user/<int:user_id>/get-recommendations", methods=['GET'])
def get_recommendations(user_id: int):
    # Let us get a user and see the top recommendations.
    user_id += 5000
    print(f"Get recs for user: {user_id}")
    hobby_ids = getUserHobbies(user_id, False)
    if len(hobby_ids) < 0:
        print("New User Detected, returning generic recommendations")
    hobby2hobby_encoded = {x: i for i, x in enumerate(hobby_ids)}
    hobby_encoded2hobby = {i: x for i, x in enumerate(hobby_ids)}

    hobbies_not_tried = [[hobby2hobby_encoded.get(x)] for x in hobby_ids]

    user_array = np.hstack(
        ([[user_id]] * len(hobbies_not_tried), hobbies_not_tried)
    )
    
    try:
        model = load_recommender_model()
    except:
        return "Model failed to load", 500
    
    ratings = model.predict(user_array).flatten()

    top_ratings_indices = ratings.argsort()[-10:][::-1]
    recommended_hobby_ids = [
        hobby_encoded2hobby.get(hobbies_not_tried[x][0]) for x in top_ratings_indices
    ]

    recommended_hobbies = getHobbiesFromIds(recommended_hobby_ids)
    recommended_hobbies_json = recommended_hobbies.to_json(orient= "records")
    parsed = json.loads(recommended_hobbies_json)
    return parsed

@nicole.route("/hobbies", methods=['POST', 'GET'])
def return_data():
    # create a dictionary
    data = {}
     
    # Open a csv reader called DictReader
    with open(MODEL_PATH + "/Data/HobbyID.csv", encoding='utf-8-sig', mode='r') as csvf:
        csvReader = csv.DictReader(csvf)
        rows = list(csvReader)
            
    return json.dumps(rows)

@nicole.route("/hobbyName", methods=['POST', 'GET'])
def return_name():
    # create a dictionary
    # data = {}
     
    # Open a csv reader called DictReader
    df = pd.read_csv('Data/HobbyID.csv')
    new = df['name']
            
    return new.to_json()
    

# @nicole.after_request
# def after_request(response):
#     response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,true')
#     response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')