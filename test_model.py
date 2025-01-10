import torch
from transformers import BertTokenizer
from rnn_model_news_analysis import BertSentimentModel
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# Load the model architecture and trained weights
output_dim = 3 
model = BertSentimentModel(output_dim)
model.load_state_dict(torch.load("bert_sentiment_model.pth", map_location="cpu"))
model.eval()  # Set the model to evaluation mode

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def preprocess_text(text, max_length=128):
    """
    Tokenizes and prepares input text for BERT model.
    """
    encoded = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return encoded["input_ids"], encoded["attention_mask"]

def predict_sentiment(text):
    """
    Predicts sentiment for a given text input.
    """
    input_ids, attention_mask = preprocess_text(text)
    with torch.no_grad():
        # Forward pass
        outputs = model(input_ids, attention_mask)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    sentiment_classes = ['Negative', 'Neutral', 'Positive']
    return sentiment_classes[predicted_class], probabilities

# Example: Test with a custom input
custom_input = "Inflation is not changing next year, it is sticky at 10%"
predicted_sentiment, probabilities = predict_sentiment(custom_input)

print(f"Input: {custom_input}")
print(f"Predicted Sentiment: {predicted_sentiment}")
print(f"Class Probabilities: {probabilities}")