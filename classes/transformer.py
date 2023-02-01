import os
import copy
import time
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import torch.nn as nn

# from classes.model import LSTM
from torch.utils.data import DataLoader
from classes.dataset import CustomDataset
from nltk.corpus import movie_reviews, subjectivity
from transformer import DistilBertForSequenceClassification
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from classes.commons import create_word_2_index, collate_fn, make_log_print, plot_data, test_single_epoch, train_single_epoch

WEIGHTS_PATH_TRANSFORMER = os.path.join('weights', 'transformer')
WEIGHTS_PATH_SUBJECTIVITY = os.path.join(WEIGHTS_PATH_TRANSFORMER, 'subjectivity_classification.pt')
WEIGHTS_PATH_POLARITY = os.path.join(WEIGHTS_PATH_TRANSFORMER, 'polarity_classification.pt')

PLOTS_PATH_TRANSFORMER = os.path.join('plots', 'transformer')
PLOTS_PATH_SUBJECTIVITY = os.path.join(PLOTS_PATH_TRANSFORMER, 'subjectivity_train_loss_accuracy_f1.png')
PLOTS_PATH_POLARITY = os.path.join(PLOTS_PATH_TRANSFORMER, 'polarity_train_loss_accuracy_f1.png')

def make_dirs():
    if not os.path.exists(WEIGHTS_PATH_TRANSFORMER):
        os.makedirs(WEIGHTS_PATH_TRANSFORMER)
    if not os.path.exists(PLOTS_PATH_TRANSFORMER):
        os.makedirs(PLOTS_PATH_TRANSFORMER)
    
def train_subjectivity_classification(epochs:int = 30, lr:float = 2e-5, device:str = 'cpu') -> nn.Module:
    """
    Do subjectivity classification using a custom classifier.
    """    
    # Get subjectivity and objectivity data
    obj = subjectivity.sents(categories='obj')
    subj = subjectivity.sents(categories='subj')

    # Compute lebels and split in train/test set
    labels = [1] * len(subj) + [0] * len(obj)

    # Split in train/test set
    dataset = CustomDataset(subj + obj, labels)
    train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=0)

    train_set_x, train_set_y = zip(*train_set)
    test_set_x, test_set_y = zip(*test_set)

    # Make train/test set
    # Since the classifier is not able to handle strings, we need to convert them to a list of integers
    # I will use the word2index dictionary to do so
    word2index = create_word_2_index(train_set_x + test_set_x)

    # Now convert the list of words to a list of integers
    train_set_x = [[word2index[word] for word in sentence] for sentence in train_set_x]
    test_set_x = [[word2index[word] for word in sentence] for sentence in test_set_x]

    # I can continue
    train_set = CustomDataset(train_set_x, train_set_y)
    test_set = CustomDataset(test_set_x, test_set_y)
    
    # Make DataLoader
    train_loader = DataLoader(train_set, batch_size=4096, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=4096, shuffle=True, collate_fn=collate_fn)
    
    # Initialize DistilBERT model
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Create variables to store the best model
    best_acc = 0
    best_f1 = 0
    best_model = None
    best_loss = np.inf

    # Create variables to store the loss, accuracy and f1 score
    base_dict = {'loss': [], 'accuracy': [], 'f1': []}
    data = {'train': copy.deepcopy(base_dict), 'test': copy.deepcopy(base_dict)}

    # Start timer
    start_time = time.time()

    # Train the model
    for epoch in range(epochs):
        # Train
        train_metrics = train_single_epoch(model, train_loader, optimizer, criterion, device)
        for key in train_metrics.keys():
            data['train'][key].append(train_metrics[key])

        # Test
        test_metrics = test_single_epoch(model, test_loader, criterion, device)
        for key in test_metrics.keys():
            data['test'][key].append(test_metrics[key])

        # Print results
        make_log_print("Train", (epoch+1, epochs), time.time() - start_time, train_metrics, test_metrics)

        # Save the best model
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
            best_f1 = test_metrics['f1']
            best_loss = test_metrics['loss']
            best_model = copy.deepcopy(model)

        # Update scheduler
        # scheduler.step()

    print()
    make_log_print("Eval", None, None, None, {'loss': best_loss, 'accuracy': best_acc, 'f1': best_f1})
    print()

    # Save the model
    best_model.save(WEIGHTS_PATH_SUBJECTIVITY)

    # Plot loss, accuracy and f1 score
    plot_data(data, title="Subjectivity train results", save_path=PLOTS_PATH_SUBJECTIVITY)
    
    return best_model

def train_polarity_classification(epochs: int = 30, lr: float = 2e-5, device: str = 'cpu'):
    """
    Do polarity classification using a trained classifier.
    """
    # Get positive and negative data
    neg = movie_reviews.paras(categories='neg')
    pos = movie_reviews.paras(categories='pos')

    # Convert list of list of list of words to list of sentences (each sentence is a list of words)
    neg = [[word for sentence in sentences for word in sentence] for sentences in neg]
    pos = [[word for sentence in sentences for word in sentence] for sentences in pos]

    # Compute lebels and split in train/test set
    labels = [1] * len(pos) + [0] * len(neg)

    # Split in train/test set
    dataset = CustomDataset(pos + neg, labels)
    train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=0)

    train_set_x, train_set_y = zip(*train_set)
    test_set_x, test_set_y = zip(*test_set)

    # Make train/test set
    # Since the classifier is not able to handle strings, we need to convert them to a list of integers
    # I will use the word2index dictionary to do so
    word2index = create_word_2_index(train_set_x + test_set_x)

    # Now convert the list of words to a list of integers
    train_set_x = [[word2index[word] for word in sentence] for sentence in train_set_x]
    test_set_x = [[word2index[word] for word in sentence] for sentence in test_set_x]

    # I can continue
    train_set = CustomDataset(train_set_x, train_set_y)
    test_set = CustomDataset(test_set_x, test_set_y)
    
    # Make DataLoader - I had to reduce the batch size to 16 because of memory issues
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=True, collate_fn=collate_fn)
    
    # Initialize DistilBERT model
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Create variables to store the best model
    best_acc = 0
    best_f1 = 0
    best_model = None
    best_loss = np.inf

    # Create variables to store the loss, accuracy and f1 score
    base_dict = {'loss': [], 'accuracy': [], 'f1': []}
    data = {'train': copy.deepcopy(base_dict), 'test': copy.deepcopy(base_dict)}
    
    # Start timer
    start_time = time.time()

    # Train the model
    for epoch in range(epochs):
        # Train
        train_metrics = train_single_epoch(model, train_loader, optimizer, criterion, device)
        for key in train_metrics.keys():
            data['train'][key].append(train_metrics[key])

        # Test
        test_metrics = test_single_epoch(model, test_loader, criterion, device)
        for key in test_metrics.keys():
            data['test'][key].append(test_metrics[key])

        # Print results
        make_log_print("Train", (epoch+1, epochs), time.time() - start_time, train_metrics, test_metrics)

        # Save the best model
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
            best_f1 = test_metrics['f1']
            best_loss = test_metrics['loss']
            best_model = copy.deepcopy(model)
        
        # Update scheduler
        # scheduler.step()
    
    print()
    make_log_print("Eval", None, None, None, {'loss': best_loss, 'accuracy': best_acc, 'f1': best_f1})
    print()

    # Save the model
    best_model.save(WEIGHTS_PATH_POLARITY)

    # Plot loss, accuracy and f1 score
    plot_data(data, title="Polarity train results", save_path=PLOTS_PATH_POLARITY)

    return best_model

def run_transformer(device: str = 'cpu'):
    # Train subjectivity classifier with custom implementation
    subj_class = train_subjectivity_classification(device=device)

    # Train polarity classifier with custom implementation
    pol_class = train_polarity_classification(device=device)