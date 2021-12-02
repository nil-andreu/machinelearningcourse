from sklearn.feature_extraction.text import HashingVectorizer
import re
import os
import pickle

current_dir = os.path.dirname(__file__)

stop_words = pickle.load(open(os.path.join(current_dir, 'pkl_object', 'stopwords.pkl'), "rb"))

def tokenizer(text):
    # Substitute some values that does not have a meaning
    text = re.sub("<[`>]*>", "", text)
    emoticons = re.findall("(?::|;|=)(?:-)?(?:\)|\(|D|P)", text.lower)

    text = re.sub("|[W]"+', ' '', text.lower()) + ' '.join(emoticons).replace('-',' ')
    tokenizer = [w for w in text.split() if w not in stop_words]
    return tokenizer

vectorizer = HashingVectorizer(decode_error='ignore',
    n_features=2**21,
    preprocessor=None,
    tokenizer=tokenizer)

