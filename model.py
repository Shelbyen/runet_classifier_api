from joblib import load

import nltk
from nltk.corpus import stopwords
from gensim.utils import tokenize
import pymorphy2
from nltk.stem import SnowballStemmer

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


def processing_text(text):
    stop_words = set(stopwords.words("russian"))
    morph = pymorphy2.MorphAnalyzer()
    snowball = SnowballStemmer(language="russian")

    filtered_tokens = []
    tokens = list(tokenize(text))
    for token in tokens:
        if token not in stop_words and "http" not in token:
            filtered_tokens.append(snowball.stem(morph.parse(token)[0].normal_form))
    return " ".join(filtered_tokens)


class Model:
    def __init__(self, path_model="runet_classifier_api/model.pkl", path_vectors="runet_classifier_api/vectorizers.pkl"):
        self.tfidf_vectorizer = load(path_vectors)
        self.clf = load(path_model)

    def get_category_prediction(self, text):
        p_text = [processing_text(text)]
        return self.clf.predict(self.tfidf_vectorizer.transform(p_text))
