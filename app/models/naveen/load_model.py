import os
from tensorflow.keras.models import load_model


DIRPATH = os.path.dirname(__file__)

def load_emotion_model():
    model = load_model(DIRPATH+'/model3.h5')
    print("Loaded EmotionClassifer")
    return model