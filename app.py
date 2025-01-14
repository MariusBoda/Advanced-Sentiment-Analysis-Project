from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer, pipeline
from newspaper import Article
from rnn_model_news_analysis import BertSentimentModel  # Assuming the model is in 'rnn_model_news_analysis.py'
import torch.nn.functional as F

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained BERT model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertSentimentModel(output_dim=3).to(device)
model.load_state_dict(torch.load("bert_sentiment_model.pth", map_location=device))
tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")

# Load the pre-trained summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Function to predict sentiment
def predict_sentiment(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        _, predicted = torch.max(outputs, dim=1)
        probabilities = F.softmax(outputs, dim=1).cpu().numpy()

    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    return sentiment_map[predicted.item()], probabilities[0]

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for sentiment prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            url = request.form['url']
            
            # Extract article content
            article = Article(url)
            article.download()
            article.parse()
            
            title = article.title
            full_text = article.text
            
            # Summarize the article
            summary = summarizer(
                full_text, 
                max_length=min(len(full_text) // 2, 500), 
                min_length=250, 
                do_sample=False
            )[0]['summary_text']
            
            # Combine title and summary for sentiment analysis
            text_to_analyze = f"{title}. {summary}"
            sentiment, probabilities = predict_sentiment(text_to_analyze)

            # Prepare probabilities for display
            sentiment_labels = ['Negative', 'Neutral', 'Positive']
            probability_values = {label: prob for label, prob in zip(sentiment_labels, probabilities)}

            return render_template(
                'result.html',
                sentiment=sentiment,
                article_url=url,
                title=title,
                summary=summary,
                probabilities=probability_values
            )
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        return render_template('error.html', error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)