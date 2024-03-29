import os
import copy
import time
import torch
import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader
from nltk.corpus import movie_reviews, subjectivity
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from classes.dataset import CustomDataset
from classes.commons import create_word_2_index, collate_fn, make_log_print, plot_data, test_single_epoch, train_single_epoch, get_basic_logger

WEIGHTS_PATH_BILSTM = os.path.join('weights', 'bilstm')
WEIGHTS_PATH_SUBJECTIVITY = os.path.join(WEIGHTS_PATH_BILSTM, 'subjectivity.pt')
WEIGHTS_PATH_POLARITY = os.path.join(WEIGHTS_PATH_BILSTM, 'polarity.pt')

PLOTS_PATH_BILSTM = os.path.join('plots', 'bilstm')
PLOTS_PATH_SUBJECTIVITY = os.path.join(PLOTS_PATH_BILSTM, 'subjectivity_loss_accuracy_f1.png')
PLOTS_PATH_POLARITY = os.path.join(PLOTS_PATH_BILSTM, 'polarity_loss_accuracy_f1.png')

logger_bilstm = get_basic_logger('BiLSTM', log_path="Log.txt")

class Attention(nn.Module):
    def __init__(self, hidden_size:int):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

        self.attn = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.attn(x)
        x = self.softmax(x)
        return x

class BiLSTM(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, emb_size:int, output_size:int = 1, n_layers:int = 2, padding_idx:int = 0, dropout_pr:float = 0.2) -> None:
        super(BiLSTM, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.output_size = output_size

        # First we need an embedding layer to convert our input tokens into vectors
        self.embedding = nn.Embedding(input_size, emb_size, padding_idx=padding_idx)

        # Then we need our memory layer, which is a LSTM in this case
        self.memory = nn.LSTM(input_size=emb_size, hidden_size=hidden_size, num_layers=n_layers, bidirectional=True, batch_first=True)
        # Can also be used with GRU, to test it out, just uncomment the line below and comment the line above
        # self.memory = nn.GRU(input_size=emb_size, hidden_size=hidden_size, num_layers=n_layers, bidirectional=True, batch_first=True)

        # Then we need a dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout_pr)

        # Then we need our attention layer
        self.attention = Attention(hidden_size * 2)

        # Then we need a classifier layer to convert our LSTM output to our desired output size
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x:torch.Tensor, lengths:torch.Tensor, heatmap:bool = False) -> torch.Tensor:
        # Embedding layer
        x = self.embedding(x)
        x = pack_padded_sequence(x, lengths.cpu().detach().numpy(), batch_first=True)

        # LSTM layer
        x, _ = self.memory(x)
        x, _ = pad_packed_sequence(x, batch_first=True)

        # Dropout layer
        x = self.dropout(x)

        if heatmap:
            return self.attention(x)
            
        # Attention layer
        x = x * self.attention(x)

        # Summing over the sequence dimension
        x = torch.sum(x, dim=1)

        # Classifier layer
        x = self.out(x)

        return x
    
    def save(self, path:str) -> None:
        logger_bilstm.info(f"Saving model to '{os.path.abspath(path)}'")
        torch.save(self.state_dict(), path)
    
    def load(self, path:str) -> None:
        logger_bilstm.info(f"Loading model from '{os.path.abspath(path)}'")
        try:
            self.load_state_dict(torch.load(path))
        except RuntimeError:
            logger_bilstm.warning("Model architecture does not match, loading only weights")
            self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

def make_dirs():
    if not os.path.exists(WEIGHTS_PATH_BILSTM):
        os.makedirs(WEIGHTS_PATH_BILSTM)
    if not os.path.exists(PLOTS_PATH_BILSTM):
        os.makedirs(PLOTS_PATH_BILSTM)
    
def train_subjectivity_classification(epochs:int = 30, lr:float = 1e-2,device:str = 'cpu') -> nn.Module:
    """
    Do subjectivity classification using a custom classifier.
    """    
    # Get subjectivity and objectivity data
    obj = subjectivity.sents(categories='obj')
    subj = subjectivity.sents(categories='subj')

    # Compute lebels and split in train/test set
    labels = [1] * len(subj) + [0] * len(obj)

    # Split in train/test set
    dataset = list(zip(subj + obj, labels))
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
    
    # Create a custom classifier
    model = BiLSTM(input_size=len(word2index), emb_size=128, hidden_size=128, output_size=1).to(device)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Scheduler (lambda scheduler)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

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
        make_log_print(logger_bilstm, "Train", (epoch+1, epochs), time.time() - start_time, train_metrics, test_metrics)

        # Save the best model
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
            best_f1 = test_metrics['f1']
            best_loss = test_metrics['loss']
            best_model = copy.deepcopy(model)

        # Update scheduler
        # scheduler.step()

    make_log_print(logger_bilstm, "Eval", None, None, None, {'loss': best_loss, 'accuracy': best_acc, 'f1': best_f1})

    # Save the model
    best_model.save(WEIGHTS_PATH_SUBJECTIVITY)

    # Plot loss, accuracy and f1 score
    plot_data(data, title="Subjectivity train results", save_path=PLOTS_PATH_SUBJECTIVITY)
    
    return best_model

def train_polarity_classification(epochs: int = 30, lr: float = 1e-3, device: str = 'cpu'):
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
    dataset = list(zip(pos + neg, labels))
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
    
    # Create a custom classifier
    model = BiLSTM(input_size=len(word2index), emb_size=128, hidden_size=128, output_size=1).to(device)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Scheduler (lambda scheduler)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.99 ** epoch)

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
        make_log_print(logger_bilstm, "Train", (epoch+1, epochs), time.time() - start_time, train_metrics, test_metrics)

        # Save the best model
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
            best_f1 = test_metrics['f1']
            best_loss = test_metrics['loss']
            best_model = copy.deepcopy(model)
        
        # Update scheduler
        # scheduler.step()
    
    make_log_print(logger_bilstm, "Eval", None, None, None, {'loss': best_loss, 'accuracy': best_acc, 'f1': best_f1})

    # Save the model
    best_model.save(WEIGHTS_PATH_POLARITY)

    # Plot loss, accuracy and f1 score
    plot_data(data, title="Polarity train results", save_path=PLOTS_PATH_POLARITY)

    return best_model

def run_bilstm(device: str = 'cpu'):
    make_dirs()
    
    # Train subjectivity classifier with custom implementation
    train_subjectivity_classification(device=device)

    # Train polarity classifier with custom implementation
    train_polarity_classification(device=device)