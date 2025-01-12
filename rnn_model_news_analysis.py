import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import kagglehub
from metaflow import FlowSpec, step, Parameter
from torch.utils.data import WeightedRandomSampler, DataLoader
from transformers import BertTokenizer, BertModel
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class BertSentimentModel(nn.Module):
    def __init__(self, output_dim):
        super(BertSentimentModel, self).__init__()
        #self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("ProsusAI/finbert")
        self.fc = nn.Linear(self.bert.config.hidden_size, output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state
        pooled_output = hidden_state[:, 0]  # Take the first token's representation
        return self.fc(pooled_output)

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, max_length, glove_embeddings=None):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_length = max_length

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if glove_embeddings is not None:
            self.embedding.weight.data.copy_(torch.tensor(glove_embeddings, dtype=torch.float32))
            self.embedding.weight.requires_grad = True  # Allow fine-tuning of embeddings

        # RNN layer
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Pass through embedding layer
        x = self.embedding(x)
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class SentimentFlow(FlowSpec):

    glove_path = Parameter("glove_path", default="glove/glove.6B.100d.txt") #download this from online
    kaggle_dataset = Parameter("kaggle_dataset", default="ankurzing/sentiment-analysis-for-financial-news")
    num_epochs = Parameter("num_epochs", 20)
    batch_size = Parameter("batch_size", 32)
    hidden_size = Parameter("hidden_size", 128)
    output_size = Parameter("output_size", 3) # (positive, negative)
    preprocessed_csv_path = "preprocessed_data.csv"  # Path to save the preprocessed data
    
    @step
    def start(self):
        print("Downloading dataset...")
        self.dataset_path = kagglehub.dataset_download(self.kaggle_dataset)
        self.file_path = os.path.join(self.dataset_path, "all-data.csv")
        
        # uncomment this if you are using the RNN model.
        #print("Loading GloVe embeddings...") 
        #self.glove_embeddings = self.load_glove_embeddings(self.glove_path)

        self.next(self.load_data)

    @step
    def load_data(self):
        columns = ["Sentiment", "News Headline"]
        self.data = pd.read_csv(self.file_path, encoding='latin-1', names=columns)
        self.data.rename(columns={"Sentiment": "label", "News Headline": "text"}, inplace=True)     

        self.next(self.preprocess)

    '''
    This preprocess step is for the BERT model.
    '''
    @step
    def preprocess(self):
        # Label encoding for the labels
        le = LabelEncoder()
        self.data['label'] = le.fit_transform(self.data['label'])

        # Load BERT tokenizer
        #tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
        self.max_length = 128

        self.max_length=128
        # Tokenize text and prepare input for BERT
        def tokenize_and_pad(text, max_length=128):
            encoded = tokenizer(
                text, 
                max_length=max_length, 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt"
            )
            return encoded["input_ids"][0], encoded["attention_mask"][0]

        # Apply tokenization and padding
        tokenized_data = self.data['text'].apply(lambda x: tokenize_and_pad(x))
        self.data['input_ids'] = tokenized_data.apply(lambda x: x[0])
        self.data['attention_mask'] = tokenized_data.apply(lambda x: x[1])

        # Convert data into PyTorch tensors
        self.data['input_ids'] = self.data['input_ids'].apply(lambda x: torch.tensor(x, dtype=torch.long))
        self.data['attention_mask'] = self.data['attention_mask'].apply(lambda x: torch.tensor(x, dtype=torch.long))

        # Save the preprocessed data to CSV
        self.data.to_csv(self.preprocessed_csv_path, index=False)
        
        # Train-test split
        self.train_data, self.test_data = train_test_split(self.data, test_size=0.25, random_state=42)

        self.next(self.prepare_data)

    '''
    This commented out preprocess step is for the RNN model. 
    Please comment out the above preprocess step and use this one if you want to test out the RNN.
    @step
    def preprocess(self):
        warnings.filterwarnings("ignore", category=FutureWarning)
        # Label encoding for the labels
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
    '''

    @step
    def prepare_data(self):
        def prepare_data(df, max_length):
            # Extract preprocessed input IDs and labels
            X = torch.stack(df['input_ids'].tolist())  # Convert the list of tensors into a single tensor
            y = torch.tensor(df['label'].values, dtype=torch.long)  # Labels remain as integers
            return X, y

        self.X_train, self.y_train = prepare_data(self.train_data, self.max_length)
        self.X_test, self.y_test = prepare_data(self.test_data, self.max_length)

        train_dataset = TensorDataset(self.X_train, self.y_train)
        test_dataset = TensorDataset(self.X_test, self.y_test)

        # Calculate class sample weights
        class_counts = np.bincount(self.train_data['label'])
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[label] for label in self.train_data['label']]

        # Create the WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights=sample_weights, 
            num_samples=len(sample_weights), 
            replacement=True
        )

        # Update DataLoader to use the sampler for oversampling
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        self.next(self.model)

    @step
    def model(self):
        output_dim = self.output_size

        self.device = 'cpu'
        self.bert_model = BertSentimentModel(output_dim).to(self.device)
        self.next(self.train)

    @step
    def train(self):
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.bert_model.parameters(), lr=0.0001)

        # Training loop
        num_epochs = self.num_epochs

        for epoch in range(num_epochs):
            self.bert_model.train()  # Set model to training mode
            running_loss = 0.0
            
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                attention_mask = inputs != 0  # Mask padding tokens
                outputs = self.bert_model(inputs, attention_mask)
                
                # Compute loss
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(self.train_loader)}")
                
        torch.save(self.bert_model.state_dict(), "bert_sentiment_model.pth")
        print("Model saved as 'bert_sentiment_model.pth'")

        self.next(self.test)

    @step
    def test(self):
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        import matplotlib.pyplot as plt

        # Evaluation loop
        self.bert_model.eval()  # Set model to evaluation mode
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                attention_mask = inputs != 0  # Mask padding tokens
                outputs = self.bert_model(inputs, attention_mask)
                
                # Get predictions
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store predictions and labels for confusion matrix
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        # Calculate accuracy
        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy}%")
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Neutral', 'Positive'])
        
        # Plot and save confusion matrix
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")
        plt.close()  # Close the plot to prevent display during runs

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