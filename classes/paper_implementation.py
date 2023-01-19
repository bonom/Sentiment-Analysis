import os
import copy
import time
import torch
import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader
from classes.dataset import CustomDataset
from classes.model import BiLSTM_CNN_Attention
from nltk.corpus import movie_reviews, subjectivity
from classes.commons import create_dataset, create_word_2_index, collate_fn, make_log_print, plot_data, test_single_epoch, train_single_epoch

WEIGHTS_PATH_PAPER = os.path.join('weights', 'paper_implementation')
WEIGHTS_PATH_SUBJECTIVITY = os.path.join(WEIGHTS_PATH_PAPER, 'subjectivity_classification.pt')
WEIGHTS_PATH_POLARITY = os.path.join(WEIGHTS_PATH_PAPER, 'polarity_classification.pt')

PLOTS_PATH = os.path.join('plots', 'paper_implementation')
PLOTS_PATH_SUBJECTIVITY = os.path.join(PLOTS_PATH, 'subjectivity_train_loss_accuracy_f1.png')
PLOTS_PATH_POLARITY = os.path.join(PLOTS_PATH, 'polarity_train_loss_accuracy_f1.png')

if not os.path.exists(WEIGHTS_PATH_PAPER):
    os.makedirs(WEIGHTS_PATH_PAPER)
if not os.path.exists(PLOTS_PATH):
    os.makedirs(PLOTS_PATH)

def train_subjectivity_classification(epochs:int = 20, lr:float = 0.001, weight_decay:float = 0.0001, device:str = 'cpu') -> nn.Module:
    """
    Do subjectivity classification using a custom classifier.
    """    
    # Get subjectivity and objectivity data
    obj = subjectivity.sents(categories='obj')
    subj = subjectivity.sents(categories='subj')

    # Compute lebels and split in train/test set
    labels = [1] * len(subj) + [0] * len(obj)
    train_set_x, train_set_y, test_set_x, test_set_y = create_dataset(subj + obj, labels)

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
    
    # Create a custom classifier
    model = BiLSTM_CNN_Attention(vocab_size=len(word2index), emb_dim=128, lstm_hidden_dim=128, cnn_num_filters=3, cnn_filter_sizes=(2,4,6), num_classes=1).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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

    make_log_print("Eval", None, None, None, {'loss': best_loss, 'accuracy': best_acc, 'f1': best_f1})

    # Save the model
    best_model.save(WEIGHTS_PATH_SUBJECTIVITY)

    # Plot loss, accuracy and f1 score
    plot_data(data, title="Subjectivity Paper Implementation", save_path=PLOTS_PATH_SUBJECTIVITY)
    
    return best_model

def train_polarity_classification(epochs: int = 20, lr: float = 0.001, weight_decay: float = 0.0, device: str = 'cpu') -> nn.Module:
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
    train_set_x, train_set_y, test_set_x, test_set_y = create_dataset(pos + neg, labels)

    # Make train/test set
    # Since the classifier is not able to handle strings, we need to convert them to a list of integers
    # I will use the word2index dictionary to do so
    word2index = create_word_2_index(train_set_x + test_set_x)

    # Now convert the list of words to a list of integers
    # train_set_x = [[[word2index[word] for word in sentence] for sentence in sentences] for sentences in train_set_x]
    # test_set_x = [[[word2index[word] for word in sentence] for sentence in sentences] for sentences in test_set_x]
    train_set_x = [[word2index[word] for word in sentence] for sentence in train_set_x]
    test_set_x = [[word2index[word] for word in sentence] for sentence in test_set_x]

    # I can continue
    train_set = CustomDataset(train_set_x, train_set_y)
    test_set = CustomDataset(test_set_x, test_set_y)
    
    # Make DataLoader - I had to reduce the batch size to 16 because of memory issues
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=True, collate_fn=collate_fn)
    
    # Create a custom classifier
    model = BiLSTM_CNN_Attention(vocab_size=len(word2index), emb_dim=128, lstm_hidden_dim=128, cnn_num_filters=3, cnn_filter_sizes=(2,4,6), num_classes=1).to(device)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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
    
    make_log_print("Eval", None, None, None, {'loss': best_loss, 'accuracy': best_acc, 'f1': best_f1})

    # Save the model
    best_model.save(WEIGHTS_PATH_POLARITY)

    # Plot loss, accuracy and f1 score
    plot_data(data, title="Polarity train results", save_path=PLOTS_PATH_POLARITY)

    return best_model