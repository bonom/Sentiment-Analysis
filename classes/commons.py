from typing import List
import nltk

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def check_downloads():
    # check if nltk data is downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    try:
        nltk.data.find('corpora/movie_reviews')
    except LookupError:
        nltk.download('movie_reviews')
    
    try:
        nltk.data.find('corpora/subjectivity')
    except LookupError:
        nltk.download('subjectivity')

    print("[OK] All required nltk data downloaded")
    return

from sklearn.model_selection import StratifiedKFold

def create_dataset(data, labels):
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for train_index, test_index in kfold.split(data, labels):
        train_set_x, test_set_x = [data[i] for i in train_index], [data[i] for i in test_index]
        train_set_y, test_set_y = [labels[i] for i in train_index], [labels[i] for i in test_index]
    
    return train_set_x, train_set_y, test_set_x, test_set_y

from nltk.stem import PorterStemmer

def stem_sentences(sentences):
    stemmer = PorterStemmer()
    stemmed_sentences = []
    for sentence in sentences:
        stemmed_words = [stemmer.stem(word) for word in sentence.split()]
        stemmed_sentences.append(' '.join(stemmed_words))
    return stemmed_sentences


def remove_stopwords(sentences:List[str]):
    # remove stopwords from a list of sentences
    stop_words = set(stopwords.words('english'))
    processed_sentences = []
    for sentence in sentences:
        processed_sentence = [word for word in sentence.split() if word.lower() not in stop_words]
        processed_sentences.append(' '.join(processed_sentence))
    return processed_sentences

def get_subjective_objective_sentences(sentences, classifier:MultinomialNB, vectorizer:CountVectorizer):
    subjective = []
    objective = []

    for sent in sentences:
        temp_subj, temp_obj = [], []
        corpus = [list2str(s) for s in sent]
        vectors = vectorizer.transform(corpus)
        predictions = classifier.predict(vectors)

        for sentence, prediction in zip(sent, predictions):
            if prediction == 1:
                temp_subj.append(list2str(sentence))
            else:
                temp_obj.append(list2str(sentence))

        subjective.append(list2str(temp_subj))
        objective.append(list2str(temp_obj))

    return subjective, objective

def lol2str(doc):
    # flatten & join
    return " ".join([w for sent in doc for w in sent])

def list2str(doc):
    # join
    return " ".join([w for w in doc])