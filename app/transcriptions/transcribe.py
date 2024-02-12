from datetime import datetime
import csv
import ffmpeg
import os
import signal
import logging
import sys
from google.cloud import speech

path = os.path.dirname(__file__).split('/')
path.pop()
path = "/".join(path)
# print(path)

sys.path.insert(1, path)  

class GracefulKiller:
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        logging.warning('gracefully exitting')
        self.kill_now = True


# RTSP stream URL
FOLDER = os.path.dirname(__file__)

client = speech.SpeechClient()

# Capture the RTSP stream and save the audio to a .wav file
def transcribing(src, model):
    if model == "SVM":
        from app.models.perry.predict import predict_svm
    elif model == "BERT":
        from app.models.perry.predict import predict_bert
    elif model == "Google":
        from app.models.perry.predict import sample_analyze_sentiment

    folderPath = FOLDER + f'/{src or "debug"}/'
    
    rtsp_stream_url = f'rtsp://localhost:8554/{src}?video=all&audio=all'
    try:
        process = (
            ffmpeg
            .input(rtsp_stream_url, rtsp_transport='tcp')  # Use TCP for transport and capture for 5 seconds
            .output("pipe:", format='wav', acodec='pcm_s16le', ac=1, ar='16000', loglevel="quiet" )  # Output format, codec, channels, rate
            .run_async(pipe_stdout=True) 
        )

        streaming_config = speech.StreamingRecognitionConfig(
            config=speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="en-US",
                enable_automatic_punctuation=True,
            )
        )

        # Define a generator that yields audio chunks from the ffmpeg process
        def audio_chunks_generator(buffer_size=4096):
            while True:
                data = process.stdout.read(buffer_size)
                if not data:
                    break
                yield data

        audio_chunks = audio_chunks_generator()

        # Streaming the audio to the Google speech service
        requests = (speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in audio_chunks)
        responses = client.streaming_recognize(streaming_config, requests)

        # Print the real-time transcriptions
        for response in responses:
            current_date = datetime.now().strftime("%d-%m-%Y")
            filename = f'{folderPath}\\{current_date}.csv'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "a", newline='') as f:

                writer = csv.writer(f)

                for result in response.results:
                    # print(f"Finished: {result.is_final}")
                    # print(f"Stability: {result.stability}")
                    alternatives = result.alternatives
                    # The alternatives are ordered from most likely to least.
                    for alternative in alternatives:
                        prediction = None
                        if model == "SVM":
                            prediction = predict_svm(alternative.transcript)
                        elif model == "BERT":
                            prediction = predict_bert(alternative.transcript)
                        elif model == "Google":
                            prediction = sample_analyze_sentiment(alternative.transcript)
                        row = [datetime.now().__str__(), alternative.transcript, prediction]
                        writer.writerow(row)
                        # print(f"{','.join(row)}")
                        # print(f"Confidence: {alternative.confidence}")
                        # print(f"Transcript: {alternative.transcript}")


    except ffmpeg.Error as e:
        print("An error occurred while capturing the RTSP stream:", e)

    except KeyboardInterrupt:
        print("Stop by signal")




def start(src, model):
    g = GracefulKiller()
    logging.warning(f'start state : {g.kill_now} src:{src} model:{model}')
    transcribing(src, model)
    logging.warning(f'end state : {g.kill_now}')

if __name__ == "__main__":
    src = sys.argv[1]
    model = sys.argv[2]
    start(src, model)
