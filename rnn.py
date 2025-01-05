from metaflow import FlowSpec, step, Parameter, current
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import kagglehub
import os
import spacy

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
    
class SentimentDataset(Dataset):
    def __init__(self, data):
        self.texts = data['text'].values
        self.labels = data['label'].values
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return torch.tensor(text, dtype=torch.long), torch.tensor(label, dtype=torch.long)


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
    num_epochs = Parameter("num_epochs", 5)
    batch_size = Parameter("batch_size", 64)
    hidden_size = Parameter("hidden_size", 128)
    output_size = Parameter("output_size", 2) # (positive, negative)
    
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
        #self.data['label'] = self.data['label'].map({0: 0, 2: 1, 4: 2})
            
        self.next(self.preprocess)

    @step
    def preprocess(self):
        """
        Preprocess the Sentiment140 dataset in batches of 10,000 and save each batch to a CSV file.
        """
        processed_file_path = 'processed_data.csv'

        # Check if preprocessed data already exists
        if os.path.exists(processed_file_path):
            print(f"Loading preprocessed data from {processed_file_path}...")
            self.data = pd.read_csv(processed_file_path)
            print(f"Loaded {len(self.data)} preprocessed texts.")
            self.next(self.tokenize_data)
            return

        print("Preprocessing data...")

        # Load spaCy model
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        stopwords = nlp.Defaults.stop_words

        # Define a list of negation words
        negation_words = {"not", "no", "never", "none", "nor", "n't"}

        # Process texts in batches of 10,000 and save to CSV
        batch_size = 10000
        total_rows = len(self.data)

        for i in range(0, total_rows, batch_size):
            batch = self.data.iloc[i:i + batch_size]
            batch_end = min(i + batch_size, total_rows)
            print(f"Processing batch {batch_end}/{total_rows}...")

            # Apply preprocessing in parallel using spaCy's pipe
            processed_texts = []
            for doc in nlp.pipe(batch['text'], batch_size=batch_size, disable=["ner", "parser"]):
                # Tokenize and remove stopwords and punctuation, but keep negations
                processed_text = " ".join([
                    token.text for token in doc 
                    if (token.text.lower() not in stopwords or token.text.lower() in negation_words) and not token.is_punct
                ])
                processed_texts.append(processed_text)

            batch['processed_text'] = processed_texts

            # Append processed batch to CSV
            batch.to_csv(processed_file_path, mode='a', header=not os.path.exists(processed_file_path), index=False)

        print(f"Finished processing {total_rows} texts.")
        self.next(self.tokenize_data)

    @step
    def tokenize_data(self):
        print("Building vocabulary from training data...")
        self.data = pd.read_csv('processed_data.csv')
        pass
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
                if i % 1000 == 0:  # print every 10 batches
                    print(f"Epoch [{epoch+1}/{self.num_epochs}], Step [{i+1}/{len(self.train_loader)}], Loss: {loss.item()}")
                
        self.next(self.test)

    @step
    def test(self):
        self.rnn_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for texts, labels in self.test_loader:
                outputs = self.rnn_model(texts)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy: {accuracy:.2f}%')

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