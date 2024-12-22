# Advanced-Sentiment-Analysis-Project
Sentiment analysis using PyTorch. Building upon my other repository regarding sentiment analysis but this project aims to use a neural network and PyTorch.

File simple_nn.ipynb uses a simple FNN (feedforward neural network). It has fully connected layers and process input data all at once and independently. Results show that this neural network is not the best in analyzing sentiment simply becacuse the FNN processes data independetly. Meaning that each word or object in the input vector is considered on its own. In otherwords, the model does not analyze the input sentence as a whole but rather looks at each individual word. This is unoptimal for input sentences which may have positive and negative words that would cancel eachother out creating a sentence with a neutral sentiment.

