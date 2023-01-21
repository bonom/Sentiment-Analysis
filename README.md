# Sentiment Analysis with PyTorch
This repository contains an implementation of a sentiment analysis model for text classification, specifically for **polarity** and **subjectivity** **classification**. The model is trained on the `nltk` *subjectivity* and *movie_reviews* datasets.
## Requirements
 - Python 3.9
 - PyTorch 1.13
 - matplotlib
 - nltk
 - numpy
 - scikit-learn

## Model

### Paper proposal
The model is composed of three main parts: a Bi-Directional LSTM, a CNN with attention mechanism, and a classifier. The LSTM is used to capture the temporal dependencies in the input sentence, while the CNN is used to capture the local dependencies in the input sentence. The attention mechanism is used to weigh the importance of each feature in the sentence. The classifier is used to make the final decision on the polarity and subjectivity of the input sentence.

### Custom proposal
The custom model consists of three main components: an Embedding layer, a Bi-directional LSTM layer, and a Linear layer for classification. The Embedding layer is used to convert the input into a vector representation, which is then passed through the LSTM layer. The LSTM layer is used to capture the contextual information in the text, and the outputs are then passed through the Linear layer for classification. Additionally, an Attention layer is also implemented to weigh the importance of different parts of the input text for the final classification. The model is trained using a combination of a CrossEntropy loss and Adam optimizer. It's main purpose is to classify a piece of text as subjective or objective.

## Getting started
 1. Clone the repository: `git clone https://github.com/<username>/sentiment-analysis-pytorch.git`
 2. Install the dependencies by running: `pip install -r requirements.txt`
 3. Run the script: `python main.py`

The script will train the following models:
 1. Baseline model using a Naive Bayes classifier
 2. Implementation of a paper proposal [1]
 3. Personal custom model
You can find the code for the models in the models.py file and the script for training them in the main.py file.

## Results
The results will be printed in the console, weights will be saved in the `weights` folder and plots of loss, accuracy and f1 score will be saved in the `plots` folder.

## Note
The datasets used for the training and testing are the nltk `subjectivity` and `movie_reviews` datasets.

## References
[1] F. Sun and N. Chu, “Text sentiment analysis based on cnn-bilstm-attention model,” in *2020 International Conference on Robots Intelligent System (ICRIS)*, 2020, pp. 749–752.