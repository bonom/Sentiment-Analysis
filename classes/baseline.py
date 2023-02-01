import numpy as np

from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import movie_reviews, subjectivity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold, cross_validate

from classes.commons import get_basic_logger, list2str, get_subjective_objective_sentences

# Function to train the subjectivity classifier
def train_subjectivity_classifier():
    """
    Train the subjectivity classifier using Stratified K-Fold with 5 splits.
    
    Returns:
    -----
    scores:
        List of scores, one for each fold
    classifier:
        Trained MultinomialNB
    vectorizer:
        Trained CountVectorizer

    Accuracy:
    -----
    0.92 +- 0.00
    """
    # init classifier and vectorizer for Polairty classification
    vectorizer = CountVectorizer()
    eval_classifier = MultinomialNB()

    # get data
    obj = subjectivity.sents(categories='obj')
    subj = subjectivity.sents(categories='subj')
    
    # preprocess dataset
    corpus = [list2str(d) for d in obj] + [list2str(d) for d in subj]
    vectors = vectorizer.fit_transform(corpus)
    targets = [0] * len(obj) + [1] * len(subj)

    # train and evaluate
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(eval_classifier, vectors, targets, cv=cv, scoring=['accuracy'])
    scores = np.array(scores['test_accuracy'])
    
    # train the classifier on the whole dataset
    classifier = MultinomialNB()
    classifier.fit(vectors, targets)

    return scores, classifier, vectorizer

# Function to train the polarity classifier
def train_polarity_classifier(subj_classifier, subj_vectorizer):
    """
    Train the polarity classifier using Stratified K-Fold with 5 splits.
    
    Args:
    -----
    subj_classifier:
        Trained MultinomialNB classifier for subjectivity classification
    subj_vectorizer:
        Trained CountVectorizer for subjectivity classification

    Returns:
    -----
    scores:
        List of scores, one for each fold
    classifier:
        Trained MultinomialNB
    vectorizer:
        Trained CountVectorizer
    
    Accuracy:
    -----
    0.84 +-0.01
    """
    # init classifier and vectorizer for Polairty classification
    vectorizer = CountVectorizer()
    eval_classifier = MultinomialNB()

    # get data
    neg = movie_reviews.paras(categories='neg')
    pos = movie_reviews.paras(categories='pos')
    
    ### Filter sentences to only include those that are classified as subjective
    # get subjectivity predictions
    corpus, _ = get_subjective_objective_sentences(neg+pos, subj_classifier, subj_vectorizer)

    # preprocess dataset
    vectors = vectorizer.fit_transform(corpus)
    targets = [0] * len(neg) + [1] * len(pos)

    # train and evaluate
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(eval_classifier, vectors, targets, cv=cv, scoring=['accuracy'])
    scores = np.array(scores['test_accuracy'])
    
    # train the classifier on the whole dataset
    classifier = MultinomialNB()
    classifier.fit(vectors, targets)

    return scores, classifier, vectorizer


def run_baseline():
    logger_baseline = get_basic_logger("Baseline")

    scores, classifier, vectorizer = train_subjectivity_classifier()
    logger_baseline.info(f"Subjectivity classifier scores: {np.mean(scores):.2f} +- {np.std(scores):.2f}")
    
    scores, classifier, vectorizer = train_polarity_classifier(classifier, vectorizer)
    logger_baseline.info(f"Polarity classifier scores: {np.mean(scores):.2f} +- {np.std(scores):.2f}")