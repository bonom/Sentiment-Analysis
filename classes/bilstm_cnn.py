import os
import copy
import time
import torch
import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader
from nltk.corpus import movie_reviews, subjectivity
from sklearn.model_selection import train_test_split

from classes.dataset import CustomDataset
from classes.commons import create_word_2_index, collate_fn, make_log_print, plot_data, test_single_epoch, train_single_epoch, get_basic_logger

WEIGHTS_PATH_BILSTM_CNN = os.path.join('weights', 'bilstm_cnn')
WEIGHTS_PATH_SUBJECTIVITY = os.path.join(WEIGHTS_PATH_BILSTM_CNN, 'subjectivity.pt')
WEIGHTS_PATH_POLARITY = os.path.join(WEIGHTS_PATH_BILSTM_CNN, 'polarity.pt')

PLOTS_PATH_BILSTM_CNN = os.path.join('plots', 'bilstm_cnn')
PLOTS_PATH_SUBJECTIVITY = os.path.join(PLOTS_PATH_BILSTM_CNN, 'subjectivity_loss_accuracy_f1.png')
PLOTS_PATH_POLARITY = os.path.join(PLOTS_PATH_BILSTM_CNN, 'polarity_loss_accuracy_f1.png')

logger_bi_lstm_cnn = get_basic_logger('BiLSTM_CNN', log_path="Log.txt")

#################################################
# Paper implementation
#################################################
class BiLSTM_CNN_Attention(nn.Module):
    def __init__(self, vocab_size, emb_dim, cnn_num_filters, cnn_filter_sizes, lstm_hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.cnn = nn.ModuleList([nn.Conv1d(in_channels=emb_dim,
                                           out_channels=cnn_num_filters,
                                           kernel_size=fs,
                                           padding=fs//2)
                                 for fs in cnn_filter_sizes])

        self.lstm = nn.LSTM(cnn_num_filters*len(cnn_filter_sizes), lstm_hidden_dim, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(2*lstm_hidden_dim, 1)
        self.fc = nn.Linear(2*lstm_hidden_dim, num_classes)
        
    def forward(self, x:torch.Tensor, lengths:torch.Tensor, heatmap: bool = False) -> torch.Tensor:
        x = self.embedding(x) # (batch_size, seq_len, emb_dim)
        x = x.permute(0, 2, 1) # (batch_size, emb_dim, seq_len)
        
        temp = []
        for conv in self.cnn:
            temp.append(nn.functional.relu(conv(x)))
        cnn_out = torch.cat(temp, dim=1) # (batch_size, cnn_num_filters*len(cnn_filter_sizes), new_seq_len)
        
        cnn_out = cnn_out.permute(0, 2, 1) # (batch_size, new_seq_len, cnn_num_filters*len(cnn_filter_sizes))
        
        cnn_out = nn.utils.rnn.pack_padded_sequence(cnn_out, lengths.cpu().detach().numpy(), batch_first=True)
        lstm_out, _ = self.lstm(cnn_out)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True) # (batch_size, seq_len, 2*lstm_hidden_dim)

        attention_weights = nn.functional.softmax(self.attention(lstm_out), dim=1) # (batch_size, seq_len, 1)

        if heatmap:
            return attention_weights

        lstm_out = lstm_out * attention_weights # (batch_size, seq_len, 2*lstm_hidden_dim)
        lstm_out = lstm_out.sum(dim=1) # (batch_size, 2*lstm_hidden_dim)
        out = self.fc(lstm_out) # (batch_size, num_classes)
        
        return out

    def save(self, path:str) -> None:
        logger_bi_lstm_cnn.info(f"Saving model to '{os.path.abspath(path)}'")
        torch.save(self.state_dict(), path)
    
    def load(self, path:str) -> None:
        logger_bi_lstm_cnn.info(f"Loading model from '{os.path.abspath(path)}'")
        try:
            self.load_state_dict(torch.load(path))
        except RuntimeError:
            logger_bi_lstm_cnn.warning("Model architecture does not match, loading only weights")
            self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

def make_dirs():
    if not os.path.exists(WEIGHTS_PATH_BILSTM_CNN):
        os.makedirs(WEIGHTS_PATH_BILSTM_CNN)
    if not os.path.exists(PLOTS_PATH_BILSTM_CNN):
        os.makedirs(PLOTS_PATH_BILSTM_CNN)

def train_subjectivity_classification(epochs:int = 30, lr:float = 1e-2, device:str = 'cpu') -> nn.Module:
    """
    Do subjectivity classification using a custom classifier.
    """    
    # Get subjectivity and objectivity data
    obj = subjectivity.sents(categories='obj')
    subj = subjectivity.sents(categories='subj')

    # Compute lebels and split in train/test set
    labels = [1] * len(subj) + [0] * len(obj)
    
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
    model = BiLSTM_CNN_Attention(vocab_size=len(word2index), emb_dim=128, lstm_hidden_dim=128, cnn_num_filters=3, cnn_filter_sizes=(2,4,6), num_classes=1).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
        make_log_print(logger_bi_lstm_cnn, "Train", (epoch+1, epochs), time.time() - start_time, train_metrics, test_metrics)

        # Save the best model
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
            best_f1 = test_metrics['f1']
            best_loss = test_metrics['loss']
            best_model = copy.deepcopy(model)

        # Update the scheduler
        # scheduler.step()

    make_log_print(logger_bi_lstm_cnn, "Eval", None, None, None, {'loss': best_loss, 'accuracy': best_acc, 'f1': best_f1})

    # Save the model
    best_model.save(WEIGHTS_PATH_SUBJECTIVITY)

    # Plot loss, accuracy and f1 score
    plot_data(data, title="Subjectivity Paper Implementation", save_path=PLOTS_PATH_SUBJECTIVITY)
    
    return best_model

def train_polarity_classification(epochs: int = 30, lr: float = 1e-3, device: str = 'cpu') -> nn.Module:
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

    dataset = list(zip(pos + neg, labels))
    train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=0)

    train_set_x, train_set_y = zip(*train_set)
    test_set_x, test_set_y = zip(*test_set)

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
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=True, collate_fn=collate_fn)
    
    # Create a custom classifier
    model = BiLSTM_CNN_Attention(vocab_size=len(word2index), emb_dim=128, lstm_hidden_dim=128, cnn_num_filters=3, cnn_filter_sizes=(2,4,6), num_classes=1).to(device)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
        make_log_print(logger_bi_lstm_cnn, "Train", (epoch+1, epochs), time.time() - start_time, train_metrics, test_metrics)

        # Save the best model
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
            best_f1 = test_metrics['f1']
            best_loss = test_metrics['loss']
            best_model = copy.deepcopy(model)
        
        # Update the scheduler
        # scheduler.step(test_metrics['loss'])

    make_log_print(logger_bi_lstm_cnn, "Eval", None, None, None, {'loss': best_loss, 'accuracy': best_acc, 'f1': best_f1})

    # Save the model
    best_model.save(WEIGHTS_PATH_POLARITY)

    # Plot loss, accuracy and f1 score
    plot_data(data, title="Polarity train results", save_path=PLOTS_PATH_POLARITY)

    return best_model

def run_bilstm_cnn(device: str = 'cpu'):
    make_dirs()

    # Train subjectivity classifier with custom implementation
    train_subjectivity_classification(device=device)

    # Train polarity classifier with custom implementation
    train_polarity_classification(device=device)