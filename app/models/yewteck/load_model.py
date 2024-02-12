import os
import keras

def load_action_model():
    MODEL_PATH = os.path.dirname(__file__)+'/last_attempt.keras'  # Ensure this path is correct
    return keras.saving.load_model(MODEL_PATH)