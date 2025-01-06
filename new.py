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
import kagglehub
from metaflow import FlowSpec, step, Parameter, current

class RNNModel(nn.Module):
    """
    Define the class for the RNN model. Metaflow steps are all after this class declaration. 
    """
    def __init__(self, input_dim, hidden_dim, output_dim, max_length):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_length = max_length
    
        # Define layers
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
            
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)  # (num_layers, batch_size, hidden_dim)
        # RNN Layer
        out, _ = self.rnn(x, h0)
        # Take the output from the last time step
        out = out[:, -1, :]
        # Fully connected layer to get output
        out = self.fc(out)
        return out


class SentimentFlow(FlowSpec):

    glove_path = Parameter("glove_path", default="glove/glove.6B.100d.txt")
    kaggle_dataset = Parameter("kaggle_dataset", default="ankurzing/sentiment-analysis-for-financial-news")
    num_epochs = Parameter("num_epochs", 5)
    batch_size = Parameter("batch_size", 64)
    hidden_size = Parameter("hidden_size", 128)
    output_size = Parameter("output_size", 3) # (positive, negative)
    
    @step
    def start(self):
        print("Downloading dataset...")
        self.dataset_path = kagglehub.dataset_download(self.kaggle_dataset)
        self.file_path = os.path.join(self.dataset_path, "all-data.csv")
        
        print("Loading GloVe embeddings...")
        
        self.glove_embeddings = self.load_glove_embeddings(self.glove_path)
        self.next(self.load_data)

    @step
    def load_data(self):
        columns = ["Sentiment", "News Headline"]
        self.data = pd.read_csv(self.file_path, encoding='latin-1', names=columns)
        self.data.rename(columns={"Sentiment": "label", "News Headline": "text"}, inplace=True)     

        self.next(self.preprocess)

    @step
    def preprocess(self):
        le = LabelEncoder()
        self.data['label'] = le.fit_transform(self.data['label'])
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

        def preprocess_text(text):
            doc = nlp(text.lower())
            # Remove stopwords and non-alphabetic words, and return lemmatized words
            return [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]

        self.data['text'] = self.data['text'].apply(preprocess_text)
        self.train_data, self.test_data = train_test_split(self.data, test_size=0.2, random_state=42)

        def get_glove_embedding(word):
            return self.glove_embeddings.get(word, np.zeros(100))

        def encode_phrase_with_glove(phrase):
            return [get_glove_embedding(word) for word in phrase]

        self.train_data['text'] = self.train_data['text'].apply(encode_phrase_with_glove)
        self.test_data['text'] = self.test_data['text'].apply(encode_phrase_with_glove)

        def pad_sequence_embeddings(seq, max_length):
            return seq + [np.zeros(100)] * (max_length - len(seq))
        
        self.max_length = max(self.data['text'].apply(len))

        self.train_data['text'] = self.train_data['text'].apply(lambda x: pad_sequence_embeddings(x, self.max_length))
        self.test_data['text'] = self.test_data['text'].apply(lambda x: pad_sequence_embeddings(x, self.max_length))

        self.next(self.prepare_data)

    @step
    def prepare_data(self):
        def prepare_data(df, max_length):
            # Pad and convert text into tensors
            X = np.array(df['text'].tolist())
            X = torch.tensor(X, dtype=torch.float32)
            
            # Labels
            y = torch.tensor(df['label'].values, dtype=torch.long)
            
            return X, y
        
        self.X_train, self.y_train = prepare_data(self.train_data, self.max_length)
        self.X_test, self.y_test = prepare_data(self.test_data, self.max_length)

        batch_size = 64
        train_dataset = TensorDataset(self.X_train, self.y_train)
        test_dataset = TensorDataset(self.X_test, self.y_test)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        self.next(self.model)

    @step
    def model(self):
        input_dim = 100  # GloVe embedding dimension
        hidden_dim = 128  # Number of hidden units in the RNN
        output_dim = 3  # Number of classes (3 in this case)
        self.device = 'cpu'

        self.rrn_model = RNNModel(input_dim, hidden_dim, output_dim, self.max_length).to(self.device)

        self.next(self.train)

    @step
    def train(self):
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.rrn_model.parameters(), lr=0.0001)

        # Training loop
        num_epochs = 10

        for epoch in range(num_epochs):
            self.rrn_model.train()  # Set model to training mode
            running_loss = 0.0
            
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.rrn_model(inputs)
                
                # Compute loss
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(self.train_loader)}")
                
        self.next(self.test)

    @step
    def test(self):
        # Evaluation loop
        self.rrn_model.eval()  # Set model to evaluation mode
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.rrn_model(inputs)
                
                # Get predictions
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy}%")

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
