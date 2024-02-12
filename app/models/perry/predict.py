import torch
from app.models.perry.load_model import load_bert_model, load_text_classifier_model

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