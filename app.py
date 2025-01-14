from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer, pipeline
from rnn_model_news_analysis import BertSentimentModel
import torch.nn.functional as F
import numpy as np
import requests

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained BERT model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertSentimentModel(output_dim=3).to(device)
model.load_state_dict(torch.load("bert_sentiment_model.pth", map_location=device))
tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")

# Load the pre-trained summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# NewsAPI key (Sign up at https://newsapi.org to get a key)
NEWS_API_URL = 'https://newsapi.org/v2/everything'

# Helper function to tokenize and split text into chunks
def tokenize_and_chunk(text, max_length=256):
    tokens = tokenizer.encode(text, truncation=True, max_length=max_length, add_special_tokens=False)
    chunks = [tokens[i:i + max_length - 2] for i in range(0, len(tokens), max_length - 2)]
    return ["[CLS] " + tokenizer.decode(chunk) + " [SEP]" for chunk in chunks]

# Function to predict sentiment for a single chunk
def predict_chunk_sentiment(chunk):
    model.eval()
    inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=256)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]

    return probabilities

# Function to determine sentiment label based on probability thresholds
def determine_sentiment_label(probabilities):
    # Assuming probabilities are ordered as [Negative, Neutral, Positive]
    negative_prob = probabilities[0]
    neutral_prob = probabilities[1]
    positive_prob = probabilities[2]

    # Determine the sentiment based on thresholds
    if positive_prob > 0.75:
        sentiment = 'Positive'
    elif 0.25 < positive_prob <= 0.75:
        sentiment = 'Somewhat Positive'
    elif neutral_prob > 0.5:
        sentiment = 'Neutral'
    elif 0.25 < negative_prob <= 0.75:
        sentiment = 'Somewhat Negative'
    else:
        sentiment = 'Negative'

    return sentiment

# Function to predict sentiment for the entire text
def predict_sentiment(text):
    chunks = tokenize_and_chunk(text)
    all_probabilities = [predict_chunk_sentiment(chunk) for chunk in chunks]

    # Aggregate probabilities (e.g., by averaging)
    aggregated_probabilities = np.mean(all_probabilities, axis=0)

    # Determine the sentiment based on the aggregated probabilities
    sentiment = determine_sentiment_label(aggregated_probabilities)

    return sentiment, aggregated_probabilities

# Function to fetch news articles using NewsAPI
def fetch_stock_news(ticker):
    params = {
        'q': ticker,
        'apiKey': NEWS_API_KEY,
        'language': 'en',
        'sortBy': 'publishedAt',
        'pageSize': 3  # Limit to 3 most recent articles
    }
    response = requests.get(NEWS_API_URL, params=params)

    if response.status_code != 200:
        raise Exception(f"Error fetching news: {response.status_code}")

    articles = response.json().get('articles', [])
    return articles

# Function to summarize the article content using the BART model
def summarize_article_content(content, length):
    summarized = summarizer(content, max_length=length, min_length=10, do_sample=False)
    return summarized[0]['summary_text'] if summarized else content

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for sentiment prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            ticker = request.form['ticker']

            # Fetch news articles related to the stock ticker
            articles = fetch_stock_news(ticker)

            if not articles:
                return render_template('error.html', error_message="No news articles found for this ticker.")

            # Extract titles and full contents of the articles
            article_details = []
            for article in articles:
                title = article['title']
                content = article['content']

                if content:
                    # Summarize the article content
                    desc = article['description'] if article['description'] else content

                    # Predict sentiment for the summarized content
                    sentiment, probabilities = predict_sentiment(content)

                    # Prepare the result for rendering
                    article_details.append({
                        'title': ''.join(summarize_article_content(title, length=10)),  # Only show the first 10 words
                        'summary': ''.join(summarize_article_content(content, length=50)),
                        'long_summary': desc,
                        'sentiment': sentiment,
                        'probabilities': probabilities
                    })

            return render_template(
                'result.html',
                ticker=ticker,
                article_details=article_details
            )

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        return render_template('error.html', error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)