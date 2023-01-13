import os
import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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
        
    def forward(self, x:torch.Tensor, lengths:torch.Tensor) -> torch.Tensor:
        x = self.embedding(x) # (batch_size, seq_len, emb_dim)
        x = x.permute(0, 2, 1) # (batch_size, emb_dim, seq_len)
        
        temp = []
        for conv in self.cnn:
            temp.append(nn.functional.relu(conv(x)))
        cnn_out = torch.cat(temp, dim=1) # (batch_size, cnn_num_filters*len(cnn_filter_sizes), new_seq_len)
        
        cnn_out = cnn_out.permute(0, 2, 1) # (batch_size, new_seq_len, cnn_num_filters*len(cnn_filter_sizes))
        
        cnn_out = nn.utils.rnn.pack_padded_sequence(cnn_out, lengths, batch_first=True)
        lstm_out, _ = self.lstm(cnn_out)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True) # (batch_size, seq_len, 2*lstm_hidden_dim)
        
        attention_weights = nn.functional.softmax(self.attention(lstm_out), dim=1) # (batch_size, seq_len, 1)
        lstm_out = lstm_out * attention_weights # (batch_size, seq_len, 2*lstm_hidden_dim)
        lstm_out = lstm_out.sum(dim=1) # (batch_size, 2*lstm_hidden_dim)
        out = self.fc(lstm_out) # (batch_size, num_classes)
        
        return out


    def save(self, path:str) -> None:
        print(f"Saving model to '{os.path.abspath(path)}'")
        torch.save(self.state_dict(), path)
    
    def load(self, path:str) -> None:
        print(f"Loading model from '{os.path.abspath(path)}'")
        try:
            self.load_state_dict(torch.load(path))
        except RuntimeError:
            print("[WARNING] Model architecture does not match, loading only weights")
            self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

#################################################
# Personal implementation
#################################################

class Attention(nn.Module):
    def __init__(self, hidden_size:int, dropout_pr:float = 0.1):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

        self.dropout = nn.Dropout(dropout_pr)
        self.attn = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.dropout(x)
        x = self.attn(x)
        x = self.softmax(x)
        return x

class LSTM(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, emb_size:int, output_size:int = 1, n_layers:int = 2, padding_idx:int = 0, dropout_pr:float = 0.5) -> None:
        super(LSTM, self).__init__()
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
        self.attention = Attention(hidden_size * 2, dropout_pr)

        # Then we need a classifier layer to convert our LSTM output to our desired output size
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x:torch.Tensor, lengths:torch.Tensor) -> torch.Tensor:
        # Embedding layer
        x = self.embedding(x)
        x = pack_padded_sequence(x, lengths, batch_first=True)

        # LSTM layer
        x, _ = self.memory(x)
        x, _ = pad_packed_sequence(x, batch_first=True)

        # Dropout layer
        x = self.dropout(x)

        # Attention layer
        x = x * self.attention(x)

        # Summing over the sequence dimension
        x = torch.sum(x, dim=1)

        # Classifier layer
        x = self.out(x)

        return x
    
    def save(self, path:str) -> None:
        print(f"Saving model to '{os.path.abspath(path)}'")
        torch.save(self.state_dict(), path)
    
    def load(self, path:str) -> None:
        print(f"Loading model from '{os.path.abspath(path)}'")
        try:
            self.load_state_dict(torch.load(path))
        except RuntimeError:
            print("[WARNING] Model architecture does not match, loading only weights")
            self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
