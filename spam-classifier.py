import nltk
import pandas as pd
import math
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
def extract_features(words):
    return {word: True for word in words}
df = pd.read_csv('emails.csv')
y=df['spam']
X=df.drop(['Email No.','spam'], axis=1)
train_test_split=0.8
last_train_index= math.floor( train_test_split*len(X))
X_train=X.iloc[:last_train_index]
X_test=X.iloc[last_train_index:]
y_train=y.iloc[:last_train_index]
y_test=y.iloc[last_train_index:]
train_data= [(extract_features(words), label) for words, label in zip(X_train, y_train)]
test_data= [(extract_features(words), label) for words, label in zip(X_test, y_test)]
classifier = NaiveBayesClassifier.train(train_data)
accuracy_score = accuracy(classifier, test_data)
