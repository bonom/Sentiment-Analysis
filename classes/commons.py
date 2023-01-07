import torch
from typing import List
import nltk
import matplotlib.pyplot as plt
import numpy as np
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
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in kfold.split(data, labels):
        train_set_x, test_set_x = [data[i] for i in train_index], [data[i] for i in test_index]
        train_set_y, test_set_y = [labels[i] for i in train_index], [labels[i] for i in test_index]
    
    return train_set_x, train_set_y, test_set_x, test_set_y

# Collate_fn
def collate_fn(batch):
    def pad_sequence(sequences:List[torch.Tensor], lengths, max_len):
        # Pad the sequences
        padded_sequences = torch.zeros(len(sequences), max_len).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_sequences[i, :end] = seq[:end]
        return padded_sequences

    # First sort the batch by the length of the sentences (in descending order)
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    # Then get the sentences and labels
    sentences, labels = zip(*batch)

    # Get the lengths of the sentences
    lengths = [len(s) for s in sentences]
    max_len = max(lengths)

    # Pad the sentences
    sentences = pad_sequence(sentences, lengths, max_len)

    # Convert the labels to a tensor
    labels = torch.stack(labels).squeeze(1)

    return sentences, labels, torch.stack([torch.tensor(l) for l in lengths])

def plot_data(loss, accuracy, f1_score, title="Loss, Accuracy & F1 Score"):
    plt.close('all')
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(np.arange(len(loss)), loss)
    ax1.set_title('Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')

    ax2.plot(np.arange(len(accuracy)), accuracy, label='Accuracy')
    ax2.plot(np.arange(len(f1_score)), f1_score, label='F1 Score')
    ax2.set_title('Accuracy & F1 Score')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy & F1 Score')
    ax2.legend()

    plt.suptitle(title)
    plt.show()

from nltk.stem import PorterStemmer

def stem_sentences(sentences):
    stemmer = PorterStemmer()
    stemmed_sentences = []
    for sentence in sentences:
        stemmed_words = [stemmer.stem(word) for word in sentence.split()]
        stemmed_sentences.append(' '.join(stemmed_words))
    return stemmed_sentences

def create_word_2_index(sentences):
    word2index = {}
    for sentence in sentences:
        for word in sentence:
            if word not in word2index:
                word2index[word] = len(word2index)
    word2index["<UNK>"] = len(word2index)

    return word2index

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