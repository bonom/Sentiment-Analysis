import os
import nltk
import torch
import logging
import torch.nn as nn
import matplotlib.pyplot as plt

from typing import List
from logging import Logger
from torch.utils.data import DataLoader
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer

#################################################
# Logger function
#################################################


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold = "\x1b[1m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = '%(asctime)s | %(levelname)s | %(name)s --> %(message)s'
    debug_format = '%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d --> %(message)s'#'%(asctime)s | %(name)s | %(filename)s:%(lineno)d  | %(message)s'

    FORMATS = {
        logging.DEBUG: bold + debug_format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + debug_format + reset,
        logging.ERROR: red + debug_format + reset,
        logging.CRITICAL: bold_red + debug_format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y/%m/%d %H:%M:%S')
        return formatter.format(record)

def get_basic_logger(name, level=logging.INFO, log_path:str=None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        logger.handlers = []

    formatter = CustomFormatter()
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s')
    fh = logging.FileHandler(log_path)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

#################################################
# Check if nltk data is downloaded
#################################################

def check_downloads() -> None:
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

    return 

#################################################
# Create dataset function
#################################################

def create_dataset(data, labels):
    # Create the dataset
    train_set_x = data[:int(len(data)*0.8)]
    train_set_y = labels[:int(len(data)*0.8)]
    test_set_x = data[int(len(data)*0.8):]
    test_set_y = labels[int(len(data)*0.8):]
    
    return train_set_x, train_set_y, test_set_x, test_set_y

#################################################
# Collate function
#################################################

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
    new_sentences = pad_sequence(sentences, lengths, max_len)

    # Convert the labels to a tensor
    labels = torch.stack(labels).squeeze(1)

    return new_sentences, labels, torch.stack([torch.tensor(l) for l in lengths])

#################################################
# Print logs
#################################################

def make_log_print(logger:Logger, status:str = "Train", epoch:tuple = None, timer:float = None, train_metrics:dict = None, test_metrics:dict = None, *args, **kwargs) -> None:
    def _convert_seconds_to_h_m_s(seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        h, m, s = int(h), int(m), int(s)

        if h > 0:
            _ret = f"{h}:{m}:{s} hh:mm:ss"
        else:
            if m > 0:
                _ret = f"{m}:{s} mm:ss"
            else:
                _ret = f"{s} seconds"
        
        return _ret
    
    _string = " " + status + " "
    if epoch is not None:
        _actual_epoch = epoch[0]
        _total_epoch = epoch[1]

        _string = " " + status + " - Epoch " + str(_actual_epoch) + "/" + str(_total_epoch) + " " 
    
    # Convert seconds of timer to minutes:seconds
    if timer is not None:
        _chrono = _convert_seconds_to_h_m_s(timer)
        
        # Compute the estimate time remaining 
        _eta = (_total_epoch - _actual_epoch) * timer/_actual_epoch 
        _eta = _convert_seconds_to_h_m_s(_eta)

    logger.info(f"{_string:=^60}")
    if train_metrics is not None:
        logger.info(f"  Training loss {train_metrics['loss']:.3f}, Training accuracy {train_metrics['accuracy']:.3f}, Training f1 {train_metrics['f1']:.3f}")
    
    if test_metrics is not None:
        logger.info(f"  Test loss {test_metrics['loss']:.3f}, Test accuracy {test_metrics['accuracy']:.3f}, Test f1 {test_metrics['f1']:.3f}")

    if timer is not None:
        logger.info(f"  Time elapsed: {_chrono} - ETA: {_eta} - Time per epoch: {timer/_actual_epoch:.2f} seconds")

    if args is not None:
        for arg in args:
            for k, v in arg.items():
                logger.info(f"  {k}: {v}")

    if kwargs is not None:
        for k, v in kwargs.items():
            logger.info(f"  {k}: {v}")
        
    _string = " End " + status.lower() + " "
    if epoch is not None:
        _string = " End " + status.lower() + " " + str(_actual_epoch) + "/" + str(_total_epoch) + " "

    logger.info(f"{_string:=^60}")

    return

#################################################
# Training and testing functions
#################################################

def train_single_epoch(model:nn.Module, train_loader: DataLoader, optimizer:torch.optim.Optimizer, criterion, device):
    model.train()
    train_loss = 0
    train_acc = 0
    train_f1 = 0
    for sentences, labels, lengths in train_loader:
        # Move to device - No need to move lengths to device since it is needed only on cpu
        sentences, labels, lengths = sentences.to(device), labels.to(device), lengths.to(device)

        # Forward pass
        predictions = model(sentences, lengths)

        # Calculate loss
        loss = criterion(predictions, labels)

        # Zero the gradients
        optimizer.zero_grad()
        # Backward pass and update weights
        loss.backward()
        optimizer.step()

        # Detach the predictions from the graph
        # torch.nn.functional.sigmoid(predictions) is deprecated
        pred = torch.sigmoid(predictions).round()
        pred = pred.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

        # Calculate accuracy and f1 score
        acc = accuracy_score(pred, labels)
        f1 = f1_score(pred, labels, zero_division=0)

        # Update train loss, accuracy and f1 score
        train_loss += loss.item()
        train_acc += acc.item()
        train_f1 += f1.item()
    
    return {
        "loss": train_loss/len(train_loader),
        "accuracy": train_acc/len(train_loader),
        "f1": train_f1/len(train_loader)
    }

def test_single_epoch(model:nn.Module, test_loader: DataLoader, criterion, device):
    model.eval()
    test_loss = 0
    test_acc = 0
    test_f1 = 0
    
    with torch.no_grad():
        for sentences, labels, lengths in test_loader:
            # Move to device - No need to move lengths to device since it is needed only on cpu
            sentences, labels, lengths = sentences.to(device), labels.to(device), lengths.to(device)

            # Forward pass
            predictions = model(sentences, lengths)

            # Calculate loss
            loss = criterion(predictions, labels)

            # Detach the predictions from the graph
            # torch.nn.functional.sigmoid(predictions) is deprecated
            pred = torch.sigmoid(predictions).round()
            pred = pred.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()

            # Calculate accuracy and f1 score
            acc = accuracy_score(pred, labels)
            f1 = f1_score(pred, labels)

            # Update val loss, accuracy and f1 score
            test_loss += loss.item()
            test_acc += acc.item()
            test_f1 += f1.item()

    return {
        "loss": test_loss/len(test_loader),
        "accuracy": test_acc/len(test_loader),
        "f1": test_f1/len(test_loader)
    }

#################################################
# Plotting functions
#################################################

def plot_data(data:dict, title="Loss, Accuracy & F1 Score", save_path:str=None):
    train_data = data['train']
    test_data = data['test']

    loss = (train_data['loss'], test_data['loss'])
    accuracy = (train_data['accuracy'], test_data['accuracy'])
    f1_score = (train_data['f1'], test_data['f1'])

    # create a subfigure with 2 rows and 3 columns
    _ , (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # plot the data
    ax1.plot(loss[0], label='Train loss')
    ax1.plot(loss[1], label='Test loss')

    ax2.plot(accuracy[0], label='Train accuracy')
    ax2.plot(accuracy[1], label='Test accuracy')

    ax3.plot(f1_score[0], label='Train f1 score')
    ax3.plot(f1_score[1], label='Test f1 score')

    plt.suptitle(title)
    if save_path is not None:
        parent = os.path.dirname(save_path)
        if not os.path.exists(parent):
            os.makedirs(parent)
        plt.savefig(save_path)
    else:
        plt.show()

#################################################
# 
#################################################

def create_word_2_index(sentences):
    word2index = {}
    for sentence in sentences:
        for word in sentence:
            if isinstance(word, list):
                for w in word:
                    if w not in word2index:
                        word2index[w] = len(word2index)
            else:
                if word not in word2index:
                    word2index[word] = len(word2index)
    word2index["<UNK>"] = len(word2index)

    return word2index

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