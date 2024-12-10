import torch
from torch import nn

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, n_layers, dropout=0.5):
        super(Decoder, self).__init__()
        
        self.embedding = nn.Embedding(output_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.leakyrelu = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden, cell):
        pass