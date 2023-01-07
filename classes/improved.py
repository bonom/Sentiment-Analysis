import os
from sklearn.metrics import accuracy_score, f1_score
import torch
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable

import matplotlib.pyplot as plt
import torch.nn as nn

from torch.utils.data import DataLoader
from nltk.corpus import movie_reviews, subjectivity
from classes.commons import create_dataset, create_word_2_index, list2str,  lol2str, collate_fn, plot_data
from classes.dataset import CustomDataset

from classes.model import BiLSTM_CNN_Attention, LSTM

def train_subjectivity_classification(epochs:int = 20, lr:float = 0.001, weight_decay:float = 0.0001, device:str = 'cpu'):
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
    # model = BiLSTM_CNN_Attention(vocab_size=len(word2index), emb_dim=100, lstm_hidden_dim=128, lstm_num_layers=1, cnn_num_filters=2, cnn_filter_sizes=(256, 256), num_classes=2)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    # criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Check if model is already trained, if so, load it and skip training
    if os.path.isdir('weights') and os.path.isfile('weights/model.pt'):
        model.load('weights/model.pt')
        print(f"[WARNING] Model already trained, skipping training")
    else:
        os.makedirs('weights', exist_ok=True)
        print(f"[WARNING] Model not trained, training it now")
        # Create variables to store the best model
        cum_loss = []
        cum_acc = []
        cum_f1 = []

        # Train the model
        tqdm_bar = tqdm(range(epochs), desc=f"Epoch 0/{epochs} - Loss: {np.inf} - Accuracy: {-np.inf} - F1: {-np.inf}")
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

            # print(f"Pred: '{y_pred}', G-Truth: '{y}', Loss: '{loss.item()}', Acc: '{acc}', F1: '{f1}'")

            tqdm_bar.set_description(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.3f} - Accuracy: {acc:.3f} - F1: {f1:.3f}")
        
        # Save the model
        model.save('weights/model.pt')

        # Plot loss, accuracy and f1 score
        plot_data(cum_loss, cum_acc, cum_f1, title="Train set")

    return 
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
    print(f"Achieved accuracy: {acc:.3f}\nAchieved f1 score: {f1:.3f}")
    
    return model

def train_polarity_classification(subjectivity_classifier: nn.Module, lr: float = 0.001, epochs: int = 10, weight_decay: float = 0.0):
    """
    Do polarity classification using a trained classifier.
    """
    return
