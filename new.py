import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import spacy
from metaflow import FlowSpec, step, Parameter, current

class RNNModel(nn.Module):
    """
    Define the class for the RNN model. Metaflow steps are all after this class declaration. 
    """
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, embedding_matrix):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32), 
            freeze=False
        )
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
            
    def forward(self, x):
        x = self.embedding(x)
        x = x.float()
        _, hidden = self.rnn(x)
        output = self.fc(hidden.squeeze(0))
        return output


class SentimentFlow(FlowSpec):

    glove_path = Parameter("glove_path", default="glove/glove.6B.100d.txt")
    kaggle_dataset = Parameter("kaggle_dataset", default="kazanova/sentiment140")
    num_epochs = Parameter("num_epochs", 5)
    batch_size = Parameter("batch_size", 64)
    hidden_size = Parameter("hidden_size", 128)
    output_size = Parameter("output_size", 2) # (positive, negative)
    
    @step
    def start(self):
        pass
        self.next(self.load_data)

    @step
    def load_data(self):
        pass      
        self.next(self.preprocess)

    @step
    def preprocess(self):
        pass
        self.next(self.tokenize_data)

    @step
    def tokenize_data(self):
        pass
        self.next(self.model)

    @step
    def model(self):
        pass
        self.next(self.train)

    @step
    def train(self):
        pass
                
        self.next(self.test)

    @step
    def test(self):
        pass

        self.next(self.end)

    @step
    def end(self):
        """
        Final step: Print summary.
        """
        print("Training complete.")
        print(f"Train data shape: {self.X_train.shape}")
        print(f"Test data shape: {self.X_test.shape}")

    @staticmethod
    def load_glove_embeddings(glove_file_path):
        """
        Helper function to load GloVe embeddings. The start step method utilizes this function.
        """
        embeddings = {}
        with open(glove_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.strip().split()
                word = values[0]
                vector = np.asarray(values[1:], dtype="float32")
                embeddings[word] = vector
        return embeddings

if __name__ == "__main__":
    SentimentFlow()
