import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

# Run once
nltk.download('punkt')
nltk.download('wordnet')

class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]