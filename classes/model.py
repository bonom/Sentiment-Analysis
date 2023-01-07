import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiLSTM_CNN_Attention(nn.Module):
    def __init__(self, vocab_size, emb_dim, lstm_hidden_dim, lstm_num_layers, cnn_num_filters, cnn_filter_sizes, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, lstm_hidden_dim, num_layers=lstm_num_layers, bidirectional=True)
        self.cnn = nn.ModuleList([nn.Conv1d(lstm_hidden_dim * 2, cnn_num_filters, kernel_size=fs) for fs in cnn_filter_sizes])
        self.attention = nn.Linear(lstm_hidden_dim * 2, 1)
        self.fc = nn.Linear(cnn_num_filters * len(cnn_filter_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x)
        # x = x.permute(1, 0, 2)
        lstm_out, _ = self.lstm(x)
        # lstm_out = lstm_out.permute(1,  0)
        conv_out = torch.cat([nn.functional.relu(conv(lstm_out)) for conv in self.cnn], dim=1)
        # x = x.permute(0, 2, 1)
        attention_out = self.attention(conv_out)
        x = torch.bmm(lstm_out, attention_out)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = nn.functional.softmax(x, dim=1)
        return x



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
    def __init__(self, input_size:int, hidden_size:int, emb_size:int, output_size:int = 1, n_layers:int = 2, padding_idx:int = 0, dropout_pr:float = 0.5):
        super(LSTM, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.output_size = output_size

        # First we need an embedding layer to convert our input tokens into vectors
        self.embedding = nn.Embedding(input_size, emb_size, padding_idx=padding_idx)

        # Then we need our LSTM layer # TODO: Test GRU
        self.memory = nn.LSTM(input_size=emb_size, hidden_size=hidden_size, num_layers=n_layers, bidirectional=True, batch_first=True)
        # self.memory = nn.GRU(input_size=emb_size, hidden_size=hidden_size, num_layers=n_layers, bidirectional=True, batch_first=True)

        # Then we need a dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout_pr)

        # Then we need our attention layer
        self.attention = Attention(hidden_size * 2, dropout_pr)

        # Then we need a classifier layer to convert our LSTM output to our desired output size
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x, lengths):
        # Embedding layer
        x = self.embedding(x)
        x = pack_padded_sequence(x, lengths.detach().numpy(), batch_first=True)

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
    