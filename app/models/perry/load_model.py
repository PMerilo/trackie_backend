import os
import pickle
import torch
from transformers import BertTokenizer
from app.models.perry.BertClassifier import BERTClassifier

DIRPATH = os.path.dirname(__file__) 

def load_text_classifier_model(filename=DIRPATH + '/svm.scikit'):
    text_classifier_labels = pickle.load(open(filename + ".labels", "rb"))
    text_classifier_model = pickle.load(open(filename, "rb"))
    print("Loaded SVM Classifier")
    return text_classifier_model, text_classifier_labels

def load_bert_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTClassifier('bert-base-uncased', 2).to(device)
    model.load_state_dict(torch.load(DIRPATH + "/bert_classifier.pth", device))
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("Loaded BertClassifier")
    return device, model, tokenizer


