import cv2
import numpy as np
from deepface import DeepFace
import os
import signal
import logging

class GracefulImageProcessor:
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        logging.warning('Gracefully exiting image processing')
        self.kill_now = True

    def detect_faces(self, frame):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces

    def analyze_emotions(self, frame, detections):
        emotions = []
        for (x, y, w, h) in detections:
            face = frame[y:y+h, x:x+w]
            result = DeepFace.analyze(face, actions=['emotion'])
            emotion = max(result['emotion'], key=result['emotion'].get)
            emotions.append(emotion)
        return emotions

    def process_frame(self, frame_data):
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        detections = self.detect_faces(frame)
        emotions = self.analyze_emotions(frame, detections)
        return emotions

    def start_processing(self):
        logging.basicConfig(level=logging.WARNING) 
        logging.warning('Starting image processing')
        while not self.kill_now:
            try:
                sample_frame = np.zeros((100, 100, 3), dtype=np.uint8)
                emotions = self.process_frame(sample_frame.tobytes())
                logging.info(f'Emotions detected: {emotions}')
            except Exception as e:
                logging.error(f'An error occurred during image processing: {e}')

        logging.warning('Received signal to stop image processing. Exiting...')

if __name__ == '__main__':
    processor = GracefulImageProcessor()
    processor.start_processing()