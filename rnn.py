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
        
        self.next(self.preprocess)

    @step
    def preprocess(self):
        """
        Optimized preprocessing step using spaCy's `pipe` for batch processing.
        Includes caching for faster re-runs.
        """
        print("Loading spaCy model...")
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])  # Disable unnecessary components
        
        # Access spaCy's built-in stop words
        stopwords = nlp.Defaults.stop_words

        # Check if cached processed data exists
        cache_file = "processed_data.csv"
        try:
            print(f"Looking for cached file at: {cache_file}")
            self.data = pd.read_csv(cache_file)
            print(f"Loaded cached data from {cache_file}. Skipping preprocessing.")
        except FileNotFoundError:
            print("No cache found. Starting preprocessing...")

            # Process text data efficiently using spaCy's pipe
            def preprocess_text(doc):
                return " ".join([token.text for token in doc if token.text.lower() not in stopwords and not token.is_punct])

            # Initialize a counter to track progress
            total_count = len(self.data)
            processed_count = 0

            # Process texts in batches
            processed_texts = []
            for doc in nlp.pipe(self.data['text'], batch_size=1000, disable=["ner", "parser"]):
                processed_texts.append(preprocess_text(doc))
                processed_count += 1

                # Print progress every 10000 texts processed
                if processed_count % 10000 == 0:
                    print(f"Processed {processed_count}/{total_count} texts...")

            # Add the processed texts to the dataframe
            self.data['processed_text'] = processed_texts

            # Save processed data to cache
            print(f"Saving processed data to {cache_file} for future use...")
            self.data.to_csv(cache_file, index=False)

        print("Preprocessing complete!")
        self.next(self.tokenize_data)

    @step
    def tokenize_data(self):
        """
        Tokenize and encode the preprocessed dataset.
        Focuses on:
        - Encoding the already preprocessed text into sequences of indices.
        - Truncated vocabulary with top 20,000 most frequent words.
        - Pre-trained GloVe embeddings for improved word representation.
        """
        print("Building vocabulary with top 20,000 most frequent words...")

        # Build vocabulary from preprocessed text
        vectorizer = CountVectorizer(
            max_features=20000,
            tokenizer=lambda x: x.split(),  # Tokenized text from `preprocess` step
            lowercase=False
        )
        vectorizer.fit(self.data['processed_text'])
        self.vocab = vectorizer.vocabulary_
        print(f"Vocabulary size after truncation: {len(self.vocab)}")

        # Encode preprocessed text as sequences of word indices
        print("Encoding text data into sequences of word indices...")
        def encode_text(tokens):
            return [self.vocab[token] for token in tokens.split() if token in self.vocab]

        self.data['encoded_text'] = self.data['processed_text'].apply(encode_text)

        # Pad sequences
        print("Padding sequences...")
        self.padded_train = pad_sequence(
            [torch.tensor(seq) for seq in self.train_data['encoded_text']],
            batch_first=True,
            padding_value=0
        )
        self.padded_test = pad_sequence(
            [torch.tensor(seq) for seq in self.test_data['encoded_text']],
            batch_first=True,
            padding_value=0
        )

        # Convert labels to tensors
        print("Converting labels to tensors...")
        self.train_labels = torch.tensor(self.train_data['label'].values, dtype=torch.long)
        self.test_labels = torch.tensor(self.test_data['label'].values, dtype=torch.long)

        # Prepare GloVe embeddings
        print("Preparing embedding matrix from GloVe embeddings...")
        embedding_dim = len(next(iter(self.glove_embeddings.values())))
        vocab_size = len(self.vocab)
        self.embedding_matrix = np.zeros((vocab_size, embedding_dim))
        for word, idx in self.vocab.items():
            if word in self.glove_embeddings:
                self.embedding_matrix[idx] = self.glove_embeddings[word]

        print(f"Embedding matrix shape: {self.embedding_matrix.shape}")

        # Compute class weights for the loss function
        print("Computing class weights for the loss function...")
        label_counts = self.train_data['label'].value_counts()
        self.class_weights = {
            label: len(self.train_data) / count for label, count in label_counts.items()
        }
        print(f"Class weights: {self.class_weights}")

        # Create DataLoaders
        print("Creating DataLoaders...")
        train_dataset = TensorDataset(self.padded_train, self.train_labels)
        test_dataset = TensorDataset(self.padded_test, self.test_labels)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        print("Tokenization and preprocessing complete.")
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