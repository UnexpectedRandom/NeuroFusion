import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, n_layers, batch_first=True, bidirectional=True)
        self.leakyReLU = nn.LeakyReLU()  # Leaky ReLU activation
        self.dropout = nn.Dropout(dropout)  # Dropout layer to prevent overfitting
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)  # Layer normalization for bidirectional output
        
    def forward(self, x):
        embedded = self.embedding(x)  # Get embeddings for input tokens
        embedded = self.dropout(embedded)  # Apply dropout to embeddings
        
        # Apply LSTM to the embedded input
        outputs, (hidden, cell) = self.lstm(embedded)
        
        # Apply Leaky ReLU activation
        outputs = self.leakyReLU(outputs)
        
        # Apply layer normalization to the LSTM outputs
        outputs = self.layer_norm(outputs)
        
        return outputs, hidden, cell  # Return outputs, hidden state, and cell state