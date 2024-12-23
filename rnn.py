from metaflow import FlowSpec, step, Parameter, current
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import kagglehub
import os

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
    """
    Parameters and hyperparameters for the RNN model. 

    GloVe data is from: https://nlp.stanford.edu/projects/glove/
    Download the glove.6B.100d.txt file from there.

    Kaggle dataset is pulled using API. 
    Consult this webpage for more information: https://www.kaggle.com/datasets/kazanova/sentiment140/data
    """
    glove_path = Parameter("glove_path", default="glove/glove.6B.100d.txt")
    kaggle_dataset = Parameter("kaggle_dataset", default="kazanova/sentiment140")
    num_epochs = Parameter("num_epochs", 10)
    batch_size = Parameter("batch_size", 8)
    hidden_size = Parameter("hidden_size", 128)
    output_size = Parameter("output_size", 3) # (positive, negative, neutral)
    
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
        if not os.path.exists(self.glove_path):
            raise FileNotFoundError(f"GloVe file not found at: {self.glove_path}")
        else:
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
        self.data['label'] = self.data['label'].map({0: 0, 2: 1, 4: 2})
        
        print("Splitting data into train and test sets...")
        self.train_data, self.test_data = train_test_split(self.data, test_size=0.2, random_state=42)
        
        self.next(self.tokenize_data)

    @step
    def tokenize_data(self):
        """
        Tokenize and create embedding matrices for train and test sets.
        """
        def create_embedding_matrix(vectorizer, glove_embeddings):
            word_to_index = {word: idx for idx, word in enumerate(vectorizer.get_feature_names_out())}
            embedding_matrix = np.zeros((len(word_to_index), 100))
            for word, idx in word_to_index.items():
                if word in glove_embeddings:
                    embedding_matrix[idx] = glove_embeddings[word]
            return embedding_matrix

        print("Fitting tokenizer on train data...")
        self.vectorizer = CountVectorizer(max_features=500, stop_words="english")
        self.X_train = self.vectorizer.fit_transform(self.train_data["text"]).toarray()
        self.embedding_matrix = create_embedding_matrix(self.vectorizer, self.glove_embeddings)
        
        print("Transforming test data...")
        self.X_test = self.vectorizer.transform(self.test_data["text"]).toarray()
        self.y_train = self.train_data["label"].values
        self.y_test = self.test_data["label"].values
        
        # Convert the X_train and X_test data into tensors
        self.X_train = torch.tensor(self.X_train, dtype=torch.long)
        self.X_test = torch.tensor(self.X_test, dtype=torch.long)
        
        # Apply padding to make sequences in X_train and X_test the same length
        self.X_train_padded = pad_sequence([torch.tensor(x) for x in self.X_train], batch_first=True, padding_value=0)
        self.X_test_padded = pad_sequence([torch.tensor(x) for x in self.X_test], batch_first=True, padding_value=0)
        
        # Make sure the y_train and y_test remain unchanged (no padding needed for labels)
        self.y_train = torch.tensor(self.y_train, dtype=torch.long)
        self.y_test = torch.tensor(self.y_test, dtype=torch.long)
        
        # Prepare the DataLoader
        train_dataset = TensorDataset(self.X_train_padded, self.y_train)
        test_dataset = TensorDataset(self.X_test_padded, self.y_test)
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        self.next(self.model)

    @step
    def model(self):
        """
        Define the RNN model.
        """
        print("Generating RNN model...")
        vocab_size, embed_size = self.embedding_matrix.shape
        self.rnn_model = RNNModel(
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            embedding_matrix=self.embedding_matrix
        )
        self.next(self.train)

    @step
    def train(self):
        """
        Train the RNN model.
        """
        print("Training the model...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.rnn_model.parameters(), lr=0.001)  
        
        for texts, labels in self.train_loader:
            print(f"Batch: texts.shape={texts.shape}, labels.shape={labels.shape}")
            break  # just to see one batch

        for epoch in range(self.num_epochs):
            self.rnn_model.train()
            running_loss = 0.0
            for i, (texts, labels) in enumerate(self.train_loader):
                # Forward pass
                outputs = self.rnn_model(texts)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                if i % 10 == 0:  # print every 10 batches
                    print(f"Epoch [{epoch+1}/{self.num_epochs}], Step [{i+1}/{len(self.train_loader)}], Loss: {loss.item()}")
                
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