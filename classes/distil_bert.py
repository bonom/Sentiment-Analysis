import os
import copy
import time
import torch
import numpy as np
import torch.nn as nn

# from classes.model import LSTM
from torch.utils.data import DataLoader
from classes.dataset import CustomDataset
from torch.nn.utils.rnn import pad_sequence
from nltk.corpus import movie_reviews, subjectivity
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, AdamW, get_linear_schedule_with_warmup
from classes.commons import create_word_2_index, make_log_print, plot_data, list2str

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

### Since model input is different from the others, we need to re-define the train and test functions

def train_single_epoch(model:nn.Module, train_loader: DataLoader, optimizer:torch.optim.Optimizer, criterion, device):
    model.train()
    train_loss = 0
    train_acc = 0
    train_f1 = 0
    for sentences, labels, _ in train_loader:
        # Move to device
        sentences, labels, _ = sentences.to(device), labels.to(device), _.to(device)

        # Forward pass
        predictions = model(sentences).logits[:,0].unsqueeze(1)

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
        for sentences, labels, _ in test_loader:
            # Move to device
            sentences, labels = sentences.to(device), labels.to(device)

            # Forward pass
            predictions = model(sentences).logits[:,0].unsqueeze(1)

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


def collate_fn(batch):
    def pad_sequence(sequences, labels, lengths, max_len):
        # Pad the sequences
        padded_sequences = torch.zeros(len(sequences), max_len).long()
        new_seq_append = []
        new_label_append = []
        for i, (seq, label) in enumerate(zip(sequences, labels)):
            seq = seq.squeeze(0)
            end = lengths[i]

            if end > max_len:
                divisors = end//max_len
                end = max_len

                for j in range(1, divisors+1):
                    _padded_seq = torch.zeros(end).long()
                    _padded_seq[:end] = seq[end*(j-1):end*j]
                    new_seq_append.append(_padded_seq)
                    new_label_append.append(label)

            padded_sequences[i, :end] = seq[:end]

        if len(new_seq_append) > 0:
            padded_sequences = torch.cat((padded_sequences, torch.stack(new_seq_append)), dim=0)
            labels = torch.cat((labels, torch.stack(new_label_append)), dim=0)
        return padded_sequences, labels

    # First sort the batch by the length of the sentences (in descending order)
    batch.sort(key=lambda x: x[0].shape[1], reverse=True)
    # Then get the sentences and labels
    sentences, labels = zip(*batch)

    # Get the lengths of the sentences
    lengths = [s.shape[1] for s in sentences]
    max_len = max(lengths)

    if max_len > 512:
        max_len = 512

    # Convert the labels to a tensor
    labels = torch.stack(labels).squeeze(1)

    # Pad the sentences
    new_sentences, new_labels = pad_sequence(sentences, labels, lengths, max_len)

    return new_sentences, new_labels, torch.stack([torch.tensor(l) for l in lengths])

    
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
    data = list(zip(subj + obj, labels))
    train_set, test_set = train_test_split(data, test_size=0.2, random_state=0)

    train_set_x, train_set_y = zip(*train_set)
    test_set_x, test_set_y = zip(*test_set)

    # Make Tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # Now convert the list of words to a list of integers
    train_set_x = [tokenizer.encode(list2str(sentence), return_tensors='pt') for sentence in train_set_x]
    test_set_x = [tokenizer.encode(list2str(sentence), return_tensors='pt') for sentence in test_set_x]

    # I can continue
    train_set = CustomDataset(train_set_x, train_set_y)
    test_set = CustomDataset(test_set_x, test_set_y)
    
    # Make DataLoader
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=True, collate_fn=collate_fn)
    
    # Initialize DistilBERT model
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
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
    dataset = list(zip(pos + neg, labels))
    train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=0)

    train_set_x, train_set_y = zip(*train_set)
    test_set_x, test_set_y = zip(*test_set)

    # Make Tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # Now convert the list of words to a list of integers
    train_set_x = [tokenizer.encode(list2str(sentence), return_tensors='pt') for sentence in train_set_x]
    test_set_x = [tokenizer.encode(list2str(sentence), return_tensors='pt') for sentence in test_set_x]

    # I can continue
    train_set = CustomDataset(train_set_x, train_set_y)
    test_set = CustomDataset(test_set_x, test_set_y)
    
    # Make DataLoader
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
    # Initialize DistilBERT model
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
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