import os
from sklearn.metrics import accuracy_score, f1_score
import torch
import numpy as np
from tqdm import tqdm

import torch.nn as nn

from torch.utils.data import DataLoader
from nltk.corpus import movie_reviews, subjectivity
from classes.commons import create_dataset, create_word_2_index, collate_fn, plot_data
from classes.dataset import CustomDataset

from classes.model import LSTM

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'weights', 'custom_implementation')
WEIGHTS_PATH_SUBJECTIVITY = os.path.join(WEIGHTS_PATH, 'subjectivity_classification.pt')
WEIGHTS_PATH_POLARITY = os.path.join(WEIGHTS_PATH, 'polarity_classification.pt')

PLOTS_PATH = os.path.join(os.path.dirname(__file__), 'plots', 'custom_implementation')
PLOTS_PATH_SUBJECTIVITY = os.path.join(PLOTS_PATH, 'subjectivity_train_loss_accuracy_f1.png')
PLOTS_PATH_POLARITY = os.path.join(PLOTS_PATH, 'polarity_train_loss_accuracy_f1.png')

def train_subjectivity_classification(epochs:int = 40, lr:float = 0.001, weight_decay:float = 0.0001, device:str = 'cpu') -> nn.Module:
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
    model = LSTM(input_size=len(word2index), emb_size=128, hidden_size=128, output_size=1).to(device)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Check if model is already trained, if so, load it and skip training
    if os.path.isfile(WEIGHTS_PATH_SUBJECTIVITY):
        model.load(WEIGHTS_PATH_SUBJECTIVITY)
        print(f"[SUBJECTIVITY] Model already trained, skipping training")
    else:
        os.makedirs(WEIGHTS_PATH, exist_ok=True)
        print(f"[SUBJECTIVITY] Model not trained, training it now")
        # Create variables to store the best model
        cum_loss = []
        cum_acc = []
        cum_f1 = []

        # Train the model
        tqdm_bar = tqdm(range(epochs), desc=f"[SUBJECTIVITY] Epoch 0/{epochs} - Loss: {np.inf} - Accuracy: {-np.inf} - F1: {-np.inf}")
        for epoch in tqdm_bar:
            # Train
            model.train()
            for x, y, l in train_loader:  
                # Move to GPU
                x = x.to(device)
                y = y.to(device)
                
                # Clear gradients          
                optimizer.zero_grad()

                # Forward propagation
                y_pred = model(x, l)

                # Compute loss and backpropagate
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()

            # y_pred can be a list of floats, so we need to round them to get accuracy and f1 score and convert them to numpy
            y_pred = torch.round(torch.sigmoid(y_pred)).cpu().detach().numpy()
            y = y.cpu().detach().numpy()

            # Compute accuracy and f1 score
            acc = accuracy_score(y, y_pred)
            f1 = f1_score(y, y_pred)

            # Store loss, accuracy and f1 score for plotting
            cum_loss.append(loss.item())
            cum_acc.append(acc)
            cum_f1.append(f1)

            tqdm_bar.set_description(f"[SUBJECTIVITY] Epoch {epoch+1}/{epochs} - Loss: {loss.item():.3f} - Accuracy: {acc:.3f} - F1: {f1:.3f}")
        
        # Save the model
        model.save('weights/subjectivity_classifier.pt')

        # Plot loss, accuracy and f1 score
        plot_data(cum_loss, cum_acc, cum_f1, title="Subjectivity train results", save_path=PLOTS_PATH_SUBJECTIVITY)

    # Test
    model.eval()
    
    with torch.no_grad():
        for x, y, l in test_loader:
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x, l)
            loss = criterion(y_pred, y)

            y_pred = torch.round(torch.sigmoid(y_pred)).cpu().detach().numpy()
            y = y.cpu().detach().numpy()

            acc = accuracy_score(y, y_pred)
            f1 = f1_score(y, y_pred)      

    # Print results
    print(f"[SUBJECTIVITY] Achieved accuracy: {acc:.3f}\n[SUBJECTIVITY] Achieved f1 score: {f1:.3f}")
    
    return model

def train_polarity_classification(epochs: int = 5, lr: float = 0.001, weight_decay: float = 0.0, device: str = 'cpu'):
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
    train_set_x = [[word2index[word] for word in sentence] for sentence in train_set_x]
    test_set_x = [[word2index[word] for word in sentence] for sentence in test_set_x]

    # I can continue
    train_set = CustomDataset(train_set_x, train_set_y)
    test_set = CustomDataset(test_set_x, test_set_y)
    
    # Make DataLoader - I had to reduce the batch size to 16 because of memory issues
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=True, collate_fn=collate_fn)
    
    # Create a custom classifier
    model = LSTM(input_size=len(word2index), emb_size=128, hidden_size=128, output_size=1).to(device)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Check if model is already trained, if so, load it and skip training
    if os.path.isdir('weights') and os.path.isfile(WEIGHTS_PATH_POLARITY):
        model.load(WEIGHTS_PATH_POLARITY)
        print(f"[POLARITY] Model already trained, skipping training")
    else:
        print(f"[POLARITY] Model not trained, training it now")
        # Create variables to store the best model
        cum_loss = []
        cum_acc = []
        cum_f1 = []

        # Train the model
        tqdm_bar = tqdm(range(epochs), desc=f"[POLARITY] Epoch 0/{epochs} - Loss: {np.inf} - Accuracy: {-np.inf} - F1: {-np.inf}")
        for epoch in tqdm_bar:
            # Train
            model.train()
            for x, y, l in train_loader:  
                # Move to GPU
                x = x.to(device)
                y = y.to(device)
                
                # Clear gradients          
                optimizer.zero_grad()

                # Forward propagation
                y_pred = model(x, l)

                # Compute loss and backpropagate
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()

                x = x.cpu().detach().numpy()
                y = y.cpu().detach().numpy()

            # y_pred can be a list of floats, so we need to round them to get accuracy and f1 score and convert them to numpy
            y_pred = torch.round(torch.sigmoid(y_pred)).cpu().detach().numpy()

            # Compute accuracy and f1 score
            acc = accuracy_score(y, y_pred)
            f1 = f1_score(y, y_pred)

            # Store loss, accuracy and f1 score for plotting
            cum_loss.append(loss.item())
            cum_acc.append(acc)
            cum_f1.append(f1)

            tqdm_bar.set_description(f"[POLARITY] Epoch {epoch+1}/{epochs} - Loss: {loss.item():.3f} - Accuracy: {acc:.3f} - F1: {f1:.3f}")

        # Save the model
        model.save(WEIGHTS_PATH_POLARITY)

        # Plot loss, accuracy and f1 score
        plot_data(cum_loss, cum_acc, cum_f1, title="Poplarity train results", save_path=PLOTS_PATH_POLARITY)

    # Test
    model.eval()

    with torch.no_grad():
        for x, y, l in test_loader:
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x, l)
            loss = criterion(y_pred, y)

            y_pred = torch.round(torch.sigmoid(y_pred)).cpu().detach().numpy()
            y = y.cpu().detach().numpy()

            acc = accuracy_score(y, y_pred)
            f1 = f1_score(y, y_pred)
    
    # Print results
    print(f"[POLARITY] Achieved accuracy: {acc:.3f}\nAchieved f1 score: {f1:.3f}")

    return model