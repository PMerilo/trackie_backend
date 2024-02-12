import torch
from app.models.perry.load_model import load_bert_model, load_text_classifier_model
from google.cloud import language_v2

text_classifier_model, text_classifier_labels = load_text_classifier_model()
device, model, tokenizer = load_bert_model()

def predict_svm(text):
    # print ("You entered: %s" % (text))
    result = text_classifier_model.predict([text])
    # print ("Classification result:")
    prediction = text_classifier_labels[int(result[0])]
    # print (response)
    return prediction

def predict_bert(text, max_length=128):
  model.eval()
  encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
  input_ids = encoding['input_ids'].to(device)
  attention_mask = encoding['attention_mask'].to(device)

  with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    _, preds = torch.max(outputs, dim=1)
    return "Positive" if preds.item() == 1 else "Negative"
  
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