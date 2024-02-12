from datetime import datetime
import csv
import ffmpeg
import os
import signal
import logging
import sys
from google.cloud import speech
from google.cloud import language_v2

path = os.path.dirname(__file__).split('/')
path.pop()
path = "/".join(path)
print(path)

sys.path.insert(1, path)  

from app.models.perry.predict import predict_bert, predict_svm

class GracefulKiller:
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        logging.warning('gracefully exitting')
        self.kill_now = True


# RTSP stream URL
rtsp_stream_url = 'rtsp://localhost:8554/phone?video=all&audio=all'
FOLDER = os.path.dirname(__file__) + '/debug/'

client = speech.SpeechClient()
# Capture the RTSP stream and save the audio to a .wav file
def transcribing(model):
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
            with open(f'{FOLDER}\\{current_date}.csv', "a", newline='') as f:

                writer = csv.writer(f)

                for result in response.results:
                    # print(f"Finished: {result.is_final}")
                    # print(f"Stability: {result.stability}")
                    alternatives = result.alternatives
                    # The alternatives are ordered from most likely to least.
                    for alternative in alternatives:
                        prediction = None
                        if model == "svm":
                            prediction = predict_svm(alternative.transcript)
                        if model == "bert":
                            prediction = predict_bert(alternative.transcript)
                        elif model == "google":
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


def sample_analyze_sentiment(text_content: str = "I am so happy and joyful.") -> None:
    """
    Analyzes Sentiment in a string.

    Args:
      text_content: The text content to analyze.
    """

    client = language_v2.LanguageServiceClient()

    # text_content = 'I am so happy and joyful.'

    # Available types: PLAIN_TEXT, HTML
    document_type_in_plain_text = language_v2.Document.Type.PLAIN_TEXT

    # Optional. If not specified, the language is automatically detected.
    # For list of supported languages:
    # https://cloud.google.com/natural-language/docs/languages
    language_code = "en"
    document = {
        "content": text_content,
        "type_": document_type_in_plain_text,
        "language_code": language_code,
    }

    # Available values: NONE, UTF8, UTF16, UTF32
    # See https://cloud.google.com/natural-language/docs/reference/rest/v2/EncodingType.
    encoding_type = language_v2.EncodingType.UTF8

    response = client.analyze_sentiment(
        request={"document": document, "encoding_type": encoding_type}
    )
    # Get overall sentiment of the input document
    # print(f"Document sentiment score: {response.document_sentiment.score}")
    # print(f"Document sentiment magnitude: {response.document_sentiment.magnitude}")
    # Get sentiment for all sentences in the document
    prediction = "Negative"
    for sentence in response.sentences:
        print(f"Using Google NLP API. S:{sentence.sentiment.score}, M:{sentence.sentiment.magnitude}")
        if sentence.sentiment.score < -0.4 and sentence.sentiment.magnitude > 0.2:
            prediction = "Positive"
    #     print(f"Sentence text: {sentence.text.content}")
    #     print(f"Sentence sentiment score: {sentence.sentiment.score}")
    #     print(f"Sentence sentiment magnitude: {sentence.sentiment.magnitude}")

    # Get the language of the text, which will be the same as
    # the language specified in the request or, if not specified,
    # the automatically-detected language.
    # print(f"Language of the text: {response.language_code}")
        
    return prediction

def start(model):
    g = GracefulKiller()
    logging.warning(f'start state : {g.kill_now}')
    transcribing(model)
    logging.warning(f'end state : {g.kill_now}')

# start()
