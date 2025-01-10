# Advanced-Sentiment-Analysis-Project

Sentiment analysis using PyTorch. Building upon my other repository regarding sentiment analysis on tweets about stocks, this project aims to test different neural networks and to teach myself PyTorch more in depth. 

## Related Work
This project is a more advanced continuation of my earlier project on sentiment analysis:  
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

## Challenges and Learnings
Throughout this project, I encountered various challenges and learning opportunities:  
1. **Model Selection**:  
   - Experimented with multiple architectures, such as FNNs, RNNs, and transformers like BERT.  
   - Learned that understanding the nature of input data is critical to choosing the right model architecture. For example, BERT's ability to capture context significantly improved results compared to simpler models like FNNs or RNNs.

2. **Performance Optimization**:  
   - Focused on hyperparameter tuning to improve model performance.  
   - Addressed overfitting issues using regularization techniques and dropout layers.  

3. **Tool Exploration**:  
   - Deepened my understanding of PyTorch for model development and training.  
   - Started learning MetaFlow for workflow management, though more work remains to integrate it fully into the project pipeline.

## Future Enhancements
- **Web Deployment**:  
  Deploy the sentiment analysis model on a user-friendly webpage using frameworks like Flask or Django.  

- **Interactive Features**:  
  Allow users to input sentences and visualize the model’s sentiment analysis predictions in real-time, accompanied by confidence scores.

- **Scalable Data Processing**:  
  Explore distributed data processing using tools like Apache Spark or Dask to handle larger datasets.

- **Additional Metrics**:  
  Introduce fine-grained sentiment analysis metrics, such as detecting sarcasm or multi-dimensional sentiment categories (e.g., joy, anger, sadness).

## Feedback and Contributions
Feedback, suggestions, and contributions are welcome! If you're interested in collaborating or have ideas for improvement, feel free to open an issue or submit a pull request.