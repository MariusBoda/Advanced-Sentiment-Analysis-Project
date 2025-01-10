# Advanced-Sentiment-Analysis-Project

Sentiment analysis using PyTorch. Building upon my other repository regarding sentiment analysis, this project aims to test different neural networks and to teach myself PyTorch more in depth. 

## Related Work
This project builds upon concepts and techniques explored in my earlier repository:  
[ML_Project_2023](https://github.com/MariusBoda/ML_Project_2023)

## Goals
- Test different neural network architectures.
- Learn more about PyTorch and MetaFlow.
- Deploy a sentiment analysis model on a webpage.
- Create a sentiment metric ranging from [0,1].

## Project Files

- **`old/simple_nn.ipynb`**:  
  Uses a simple FNN (feedforward neural network). It has fully connected layers and processes input data all at once and independently. Results show that this neural network is not optimal for analyzing sentiment because the FNN processes data independently. Each word or object in the input vector is considered on its own. In other words, the model does not analyze the input sentence as a whole but rather looks at each individual word. This is suboptimal for input sentences that may have both positive and negative words, which would cancel each other out, creating a sentence with a neutral sentiment.

- **`rnn_model_news_analysis.py`**:  
  Contains both RNN and BERT models.  
  - **RNN Results**: Test accuracy was around 60%, and the confusion matrix was not very exciting.  
  - **BERT Results**: Showed a significant increase in test accuracy, achieving around 85%. The confusion matrix for the BERT model is shown below:

  ![Confusion Matrix](confusion_matrix.png)

