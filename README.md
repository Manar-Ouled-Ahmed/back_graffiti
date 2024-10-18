# Sentiment Classification with ETL Pipeline

This Google Colaboratory Notebook (`sentiment_classification.ipynb`) contains a sentiment classification project using machine learning techniques with an ETL (Extract, Transform, Load) pipeline approach.

## Objective
The objective of this project is to build a sentiment classification model using machine learning techniques, with a focus on implementing an ETL pipeline for efficient data processing.

## ETL Pipeline

### Extract: Load the Dataset
The dataset contains various columns with information related to tweets and sentiment analysis. Loading the dataset is the first step in the ETL pipeline to understand the data and prepare it for analysis.

### Transform: Data Preprocessing
- The text data is preprocessed using NLTK library and regular expressions to remove stopwords and perform feature extraction. This step helps clean and transform the text data into a format suitable for machine learning models.

### Load: Model Training
- The dataset is split into training and testing sets. This step ensures that the model is trained on a subset of the data and evaluated on unseen data to measure its generalization performance.
- Three classifiers are trained: Multi-Layer Perceptron (MLP), Support Vector Machine (SVM), and Naive Bayes. Training multiple models helps identify the most suitable algorithm for sentiment classification.

## Evaluation
- Evaluation metrics such as confusion matrix and classification report are used to assess the performance of each model. These metrics provide insights into the model's accuracy, precision, recall, and F1 score, helping to understand its strengths and weaknesses.

## Packages Import

The necessary packages are imported for data preprocessing and model training:
- `re`: Regular expression library for text preprocessing. Used to clean and preprocess text data for better model performance.
- `nltk`: Natural Language Toolkit for text processing. Utilized for text tokenization, stopwords removal, and stemming.
- `pandas`: Data manipulation library. Essential for handling and analyzing structured data.
- `TfidfVectorizer`: To convert text data into numerical features. Helps transform text data into a format suitable for machine learning models.
- `train_test_split`: For splitting the dataset into training and testing sets. Crucial for training and evaluating the model.
- `MLPClassifier`, `SVC`, `GaussianNB`: Machine learning classifiers used. Different classifiers are tested to find the best model for sentiment classification.
- `confusion_matrix`, `classification_report`: Evaluation metrics. Used to assess the model's performance and understand its predictive capabilities.
  
## Business & Data understanding

To begin the analysis, the dataset is loaded from the CSV file `Tweets.csv` using the following Python code:

```python
import pandas as pd

# Load the dataset from the CSV file
df = pd.read_csv("/content/Tweets.csv")

# Display the first few rows of the dataset
df.head()
This code snippet imports the necessary libraries, loads the data, and shows the initial rows of the dataset for a quick understanding of its contents.
For detailed implementation and code, refer to the `sentiment_classification.ipynb` file.
