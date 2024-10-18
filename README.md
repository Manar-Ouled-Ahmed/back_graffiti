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
```
This code snippet imports the necessary libraries, loads the data, and shows the initial rows of the dataset for a quick understanding of its contents.
The dataset contains the following columns:

- `tweet_id`: Unique ID for each tweet
- `airline_sentiment`: Sentiment of the tweet (positive, neutral, negative)
- `airline_sentiment_confidence`: Confidence level of the sentiment classification
- `negativereason`: Reason for negative sentiment (if applicable)
- `negativereason_confidence`: Confidence level of the negative sentiment reason
- `airline`: Airline associated with the tweet
- `airline_sentiment_gold`: Gold standard sentiment for airline sentiment analysis
- `name`: Twitter handle of the user
- `negativereason_gold`: Gold standard reason for negative sentiment
- `retweet_count`: Number of retweets
- `text`: Tweet content
- `tweet_coord`: Coordinates of the tweet
- `tweet_created`: Date and time of the tweet
- `tweet_location`: Location mentioned in the tweet
- `user_timezone`: Timezone of the user
This description provides a clear overview of the columns present in the dataset, making it easier for users to understand the data structure before working with it.
# Data Overview 
The `df.info()` output for the DataFrame with 14640 entries across 15 columns highlighted the presence of NaN values in various columns like 'negativereason' and 'tweet_location'. These missing values underscore the critical need for data cleaning. Data cleaning is essential for handling missing data, ensuring data accuracy, and improving the reliability of analyses. By addressing these NaN values through appropriate techniques such as imputation or removal, data quality is enhanced, paving the way for more accurate and insightful data analysis and decision-making processes.
# Airline Sentiment Distribution 
The output from `df.airline_sentiment.value_counts()` unveils the distribution of sentiments within the dataset, showcasing 9178 instances of 'negative', 3099 of 'neutral', and 2363 of 'positive' sentiments. This breakdown provides valuable insights into the overall sentiment landscape of the dataset. Understanding these sentiment distributions is crucial for analyzing customer feedback effectively, identifying trends, and guiding decision-making processes within the realm of sentiment analysis and customer satisfaction evaluation.
## Data Preparation
Feature and Label Extraction
The commands `features=df.iloc[:,10]` and `labels=df.iloc[:,1]` extract features and labels from the DataFrame, focusing on the 10th and 1st columns, respectively. This separation is crucial for machine learning tasks, where features represent input data influencing predictions, and labels are the target outputs to be predicted. This structured extraction process sets the stage for model training, evaluation, and predictive analysis within machine learning workflows.
## Data Cleaning
The following code snippet demonstrates text data preprocessing to clean and standardize text entries for analysis tasks:

```python
tidy_features = []
for i in range(len(features)):
    tmp = re.sub(r'[^a-zA-Z]', ' ', features[i])
    tmp = re.sub(r'\s[a-zA-Z]\s', ' ', tmp)
    tmp = re.sub(r'\s+', ' ', tmp)
    tmp = tmp.lower()
    tidy_features.append(tmp)
```
This code iterates through the features, removing non-alphabetic characters, single-letter words, and extra whitespaces from each text entry. The resulting text is converted to lowercase and stored in the tidy_features list. Displayed are a few original feature entries before the cleaning process. This preprocessing step is crucial for enhancing text data quality and ensuring consistency in subsequent analysis tasks.
## Word Embedding
To vectorize text data using TF-IDF, the code snippet below first downloads the English stopwords from NLTK. It then utilizes the TfidfVectorizer to transform the preprocessed text data stored in `tidy_features` into numerical features represented by a TF-IDF matrix.

```python
import nltk
nltk.download('stopwords')

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=2000, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(tidy_features).toarray()

# Analyzing the output
print(X)
```
Upon running this code, it is normal to observe a sparse array with many elements being 0 values. This is expected in the context of text classification tasks, where the TF-IDF matrix represents the frequency-weighted presence of terms in documents. The presence of zeros indicates that certain terms from the entire vocabulary may not appear in specific documents, which is a common occurrence in text classification scenarios. The TF-IDF vectorization step is essential for converting raw text data into a format suitable for machine learning models, enabling effective text analysis and classification tasks.
## Data Split
The code snippet below demonstrates the process of splitting the data into training and testing sets using the `train_test_split` function from scikit-learn. This step is crucial in machine learning workflows to evaluate the performance of models on unseen data.

```python
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)
```
By splitting the data into training and testing subsets, with 80% of the data assigned for training (X_train and y_train) and 20% for testing (X_test and y_test), this division allows for model training on one portion of the data and evaluation on another, enabling assessment of the model's generalization performance on unseen data. This practice is essential for validating the effectiveness and robustness of machine learning models before deployment.
## Machine Learning: NB Vs SVM Vs Neural Network
The code snippet below showcases the initialization of various classification models for the purpose of conducting a comparative analysis. These models include a Gaussian Naive Bayes classifier, Linear Support Vector Machine (SVM), RBF Support Vector Machine, Sigmoid Support Vector Machine, Polynomial Support Vector Machine with degree 2, and a Neural Network classifier. 

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Initialize classifiers for comparative analysis
gnb = GaussianNB()  # Gaussian Naive Bayes classifier
linear_svm = SVC(kernel='linear')  # Linear Support Vector Machine
rbf_svm = SVC(kernel='rbf')  # RBF Support Vector Machine
sigmoid_svm = SVC(kernel='sigmoid')  # Sigmoid Support Vector Machine
poly_svm = SVC(kernel='poly', degree=2)  # Polynomial Support Vector Machine with degree=2
neural = MLPClassifier(hidden_layer_sizes=(100, 20), activation='logistic', solver='adam')  # Neural Network classifier
```
We aim to compare the performance of these models on our dataset
By preparing these classifiers, the intention is to evaluate and compare their performance on the dataset under consideration. This comparative analysis will provide insights into how each model handles the data and its predictive capabilities, aiding in the selection of the most suitable model for the classification task at hand.
In the subsequent code snippet, the initialized classification models, including Gaussian Naive Bayes, SVMs with various kernels, and a Neural Network, are trained on the training data (`X_train` and `y_train`) to learn patterns and relationships within the dataset for subsequent prediction tasks.

```python
# Train the classification models
gnb.fit(X_train, y_train)  # Train Gaussian Naive Bayes classifier
linear_svm.fit(X_train, y_train)  # Train Linear SVM
rbf_svm.fit(X_train, y_train)  # Train RBF SVM
sigmoid_svm.fit(X_train, y_train)  # Train Sigmoid SVM
ploy_svm.fit(X_train, y_train)  # Train Polynomial SVM
neural.fit(X_train, y_train)  # Train Neural Network to find the best weight matrix
```
By fitting these models on the training data, each algorithm adapts its internal parameters to the training set, aiming to capture the underlying patterns and structures essential for making accurate predictions. This training phase is fundamental in the machine learning workflow, enabling models to learn from the provided data and improve their predictive performance.
The following code snippet demonstrates the process of using the trained classification models to predict labels for the test data (`X_test`). Each model, including Gaussian Naive Bayes, SVMs with different kernels, and the Neural Network, is used to predict the labels based on the features in the test set.

```python
# Make predictions on the test data
y_nb = gnb.predict(X_test)  # Predict using Gaussian Naive Bayes
y_linear_svm = linear_svm.predict(X_test)  # Predict using Linear SVM
y_rbf_svm = rbf_svm.predict(X_test)  # Predict using RBF SVM
y_ploy_svm = ploy_svm.predict(X_test)  # Predict using Polynomial SVM
y_sigmoid_svm = sigmoid_svm.predict(X_test)  # Predict using Sigmoid SVM
y_neural = neural.predict(X_test)  # Predict using Neural Network
```
By applying these trained models to the test data, predictions are generated for each model, enabling an assessment of their performance on unseen data. This step is crucial for evaluating how well the models generalize to new instances and for comparing their predictive capabilities in the context of the specific classification task.
## Performance Evaluation 
### Naive Bayes classifier
In the evaluation results for the Naive Bayes classifier, the confusion matrix reveals its performance across sentiment classes. Notably, out of 1836 instances labeled as "negative," the model correctly predicted 552. For "neutral," it had 189 correct predictions out of 617 instances, and for "positive," 373 out of 475 were predicted accurately.
The precision, recall, and F1-score metrics further elucidate the classifier's performance. Precision scores of 0.88, 0.26, and 0.24 were observed for the "negative," "neutral," and "positive" classes, respectively. These values indicate the accuracy of positive predictions within each class. The recall values, representing the proportion of actual positives correctly predicted, were 0.30, 0.31, and 0.79 for the respective classes.
Balancing precision and recall, the F1-scores contribute to the overall accuracy of 0.38 on the 2928 instances tested.
### Linear Support Vector Machine (SVM)
In the evaluation results for the Linear Support Vector Machine (SVM) classifier, the confusion matrix illustrates that out of 1836 instances labeled as "negative," 1683 were correctly predicted, 321 instances labeled as "neutral" had 321 correct predictions, and 298 instances labeled as "positive" were predicted accurately. The classification report reveals precision scores of 0.82, 0.64, and 0.78 for the "negative," "neutral," and "positive" classes, respectively, indicating the accuracy of positive predictions within each class. The recall values for the classes are 0.92, 0.52, and 0.63, showing the proportion of actual positives that were correctly predicted. The F1-scores strike a balance between precision and recall, resulting in an overall accuracy of 0.79 on the 2928 instances tested.
### RBF SVM classifier
In the evaluation results for the RBF SVM classifier, the confusion matrix showcases its performance across sentiment classes. Notably, out of 1836 instances labeled as "negative," the model correctly predicted 1748. For "neutral," it had 280 correct predictions out of 617 instances, and for "positive," 285 out of 475 were predicted accurately. The precision values for "negative," "neutral," and "positive" classes were 0.80, 0.72, and 0.80, respectively, indicating the accuracy of positive predictions within each class. In terms of recall, the classifier achieved scores of 0.95, 0.45, and 0.60 for the respective sentiment categories. Balancing precision and recall, the F1-scores contribute to the overall accuracy of 0.79 on the 2928 instances tested
### Sigmoid SVM classifier
In the performance evaluation of the Sigmoid SVM classifier, the confusion matrix provides a detailed view of its performance across sentiment classes. Notably, out of 1836 instances labeled as "negative," the model accurately predicted 1686. For "neutral," it correctly classified 310 out of 617 instances, and for "positive," 289 out of 475 were predicted accurately.
Precision, recall, and F1-score metrics further elucidate the classifier's performance. Precision values of 0.81, 0.64, and 0.78 were observed for the "negative," "neutral," and "positive" classes, respectively, indicating the accuracy of positive predictions within each class. The recall values, representing the proportion of actual positives correctly predicted, were 0.92, 0.50, and 0.61 for the respective classes.
Balancing precision and recall, the F1-scores contribute to the overall accuracy of 0.78 on the 2928 instances tested
### Polynomial (degree 2) SVM classifier
In the evaluation results for the Polynomial (degree 2) SVM classifier, the confusion matrix provides insights into its performance across sentiment classes. Notably, out of 1836 instances labeled as "negative," the model accurately predicted 1765. For "neutral," it correctly classified 253 out of 617 instances, and for "positive," 255 out of 475 were predicted correctly.
Precision, recall, and F1-score metrics further detail the classifier's performance. Precision values of 0.78, 0.70, and 0.84 were observed for the "negative," "neutral," and "positive" classes, respectively, indicating the accuracy of positive predictions within each class. The corresponding recall values, reflecting the proportion of actual positives correctly predicted, stood at 0.96, 0.41, and 0.54 for the respective sentiment categories.
Balancing precision and recall, the F1-scores contribute to the overall accuracy of 0.78 on the 2928 instances tested.
### Neural Network classifier
In the performance evaluation of the Neural Network classifier, the confusion matrix illustrates its performance across sentiment classes. Notably, out of 1836 instances labeled as "negative," the model correctly predicted 1535. For "neutral," it made 342 correct predictions out of 617 instances, and for "positive," 302 out of 475 were predicted accurately.
Further insights into the classifier's performance are provided by the precision, recall, and F1-score metrics. Precision scores of 0.85, 0.55, and 0.62 were observed for the "negative," "neutral," and "positive" classes, respectively, indicating the accuracy of positive predictions within each class. The recall values, representing the proportion of actual positives correctly predicted, were 0.84, 0.55, and 0.64 for the respective classes.
Balancing precision and recall, the F1-scores contribute to the overall accuracy of 0.74 on the 2928 instances tested.

Based on the evaluation results, the Polynomial (degree 2) SVM emerges as the most suitable model for sentiment classification. It excels in accurately predicting negative sentiments while maintaining a respectable balance in classifying neutral and positive sentiments. With a high recall for negative sentiments and a reasonable overall accuracy of 0.78, this model demonstrates a strong potential for effectively categorizing sentiment classes. 

## Storing Trained Model and Vectorizer
To facilitate model reuse and deployment, the provided Python code snippet utilizes the pickle module to serialize and save the trained vectorizer and neural network model objects to disk. This approach enables you to store these objects in a persistent format, allowing for quick loading without the need for retraining or recreating the vectorizer from scratch. By saving these objects using pickle, you ensure easy sharing and distribution of the trained model and vectorizer, promoting collaboration and reproducibility in machine learning projects. The serialized objects are stored in files named iasria_vect.pickle for the vectorizer and iasria_model.pickle for the neural network model.

For detailed implementation and code, refer to the `sentiment_classification.ipynb` file.
