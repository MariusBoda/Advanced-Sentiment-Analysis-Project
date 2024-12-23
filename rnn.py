from metaflow import FlowSpec, step, Parameter, current
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import kagglehub
import os


class SentimentFlow(FlowSpec):
    """
    Parameters and hyperparameters for the RNN model. 

    GloVe data is from: https://nlp.stanford.edu/projects/glove/
    Download the glove.6B.100d.txt file from there.

    Kaggle dataset is pulled using API. 
    Consult this webpage for more information: https://www.kaggle.com/datasets/kazanova/sentiment140/data
    """
    glove_path = Parameter("glove_path", default="glove/glove.6B.100d.txt")
    kaggle_dataset = Parameter("kaggle_dataset", default="kazanova/sentiment140")
    
    batch_size = Parameter("batch_size", default=32)
    epochs = Parameter("epochs", default=10)

    embed_size = Parameter("embed_size", 128)
    hidden_size = Parameter("hidden_size", 128)
    output_size = Parameter("output_size", 3)
    
    @step
    def start(self):
        """
        Initial step: Setup or load external resources.
        """
        print("Downloading dataset...")
        self.dataset_path = kagglehub.dataset_download(self.kaggle_dataset)
        print(f"Dataset downloaded at: {self.dataset_path}")
        self.file_path = os.path.join(self.dataset_path, "training.1600000.processed.noemoticon.csv")
        
        print("Loading GloVe embeddings...")
        self.glove_embeddings = self.load_glove_embeddings(self.glove_path)
        
        self.next(self.load_data)

    @step
    def load_data(self):
        """
        Load and preprocess the Sentiment140 dataset.
        """
        columns = ['target', 'id', 'date', 'flag', 'user', 'text']
        full_data = pd.read_csv(self.file_path, encoding="latin-1", names=columns)
        self.data = full_data[['target', 'text']].rename(columns={"target": "label"})
        
        print("Splitting data into train and test sets...")
        self.train_data, self.test_data = train_test_split(self.data, test_size=0.2, random_state=42)
        
        self.next(self.tokenize_data)

    @step
    def tokenize_data(self):
        """
        Tokenize and create embedding matrices for train and test sets.
        """
        def tokenize_and_create_embeddings(texts, glove_embeddings, max_features=1000):
            vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
            X = vectorizer.fit_transform(texts).toarray()
            word_to_index = {word: idx for idx, word in enumerate(vectorizer.get_feature_names_out())}
            embedding_matrix = np.zeros((len(word_to_index), 100))
            for word, idx in word_to_index.items():
                if word in glove_embeddings:
                    embedding_matrix[idx] = glove_embeddings[word]
            return X, word_to_index, embedding_matrix
        
        print("Processing train data...")
        self.X_train, self.word_to_index, self.embedding_matrix = tokenize_and_create_embeddings(
            self.train_data["text"], self.glove_embeddings
        )
        
        print("Processing test data...")
        self.X_test, _, _ = tokenize_and_create_embeddings(
            self.test_data["text"], self.glove_embeddings
        )
        
        self.next(self.model)

    @step
    def model(self):
        """
        Define the RNN model.
        """
        print("Generating RNN model.")


        self.next(self.train)

    @step
    def train(self):
        """
        Train the model.
        """
        print("Training RNN model.")


        self.next(self.end)

    @step
    def end(self):
        """
        Final step: Wrap up and summarize results.
        """
        print("Data preprocessing complete!")
        print(f"Train data shape: {self.X_train.shape}")
        print(f"Test data shape: {self.X_test.shape}")
        print("Ready for training a model.")

    @staticmethod
    def load_glove_embeddings(glove_file_path, embedding_dim=100):
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
    
    @staticmethod
    def define_neural(input_size, hidden_size, output_size, embedding_matrix, num_layers=1, dropout=0.5):
        """
        Helper function to define a RNN model with PyTorch.

        Args:
            input_size (int): Size of the input features (embedding dimension).
            hidden_size (int): Number of hidden units in the RNN.
            output_size (int): Number of output classes (e.g., 2 for binary classification).
            embedding_matrix (np.ndarray): Pretrained embedding matrix.
            num_layers (int): Number of RNN layers.
            dropout (float): Dropout probability.

        Returns:
            torch.nn.Module: A PyTorch RNN model.

        """
        class RNNModel(nn.Module):
            def __init__(self, input_size, hidden_size, output_size, embedding_matrix, num_layers, dropout):
                super(RNNModel, self).__init__()
                vocab_size, embedding_dim = embedding_matrix.shape 
                self.embedding = nn.Embedding(vocab_size, embedding_dim)

                self.rnn = nn.RNN(input_size, hidden_size, output_size, embedding_matrix, num_layers, dropout)

                self.fc = nn.Linear(hidden_size, output_size)
            
            def forward(self, x):
                x = self.embedding(x)
                h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
                out, _ = self.rnn(x, h0)
                out = self.fc(out[:, -1, :])
                return out

        return RNNModel(input_size, hidden_size, output_size, embedding_matrix, num_layers, dropout)
    


if __name__ == "__main__":
    SentimentFlow()