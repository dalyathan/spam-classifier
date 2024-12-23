import nltk
import pandas as pd
import math
import numpy as np
from nltk.classify import NaiveBayesClassifier

def read_data():
    df = pd.read_csv('emails.csv')
    shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    y=shuffled_df['Prediction']
    X=shuffled_df.drop(['Email No.','Prediction'], axis=1)
    X_std=(X - X.mean()) / X.std()
    return X_std,y

def split_data(X,y,train_test_split=0.7):
    last_train_index= math.floor( train_test_split*len(X))
    X_train=X.iloc[:last_train_index]
    y_train=y.iloc[:last_train_index]
    X_test=X.iloc[last_train_index:]
    y_test=y.iloc[last_train_index:]
    return  X_train.to_dict(orient='records'),X_test.to_dict(orient='records'),y_train,y_test

def train_and_predict(X_train,X_test,y_train):
    train_data= zip(X_train, y_train)
    classifier = NaiveBayesClassifier.train(train_data)
    predicted= [classifier.classify(data) for data in X_test]
    return predicted

def true_positive(predicted, y_test):
    TP=0
    for p, y in zip(predicted, y_test):
        if p == 1 and y == 1:
            TP += 1
    return TP

def true_negative(predicted, y_test):
    TN=0
    for p, y in zip(predicted, y_test):
        if p == 0 and y == 0:
            TN += 1
    return TN

def false_positive(predicted, y_test):
    FP=0
    for p, y in zip(predicted, y_test):
        if p == 1 and y == 0:
            FP += 1
    return FP

def false_negative(predicted, y_test):
    FN=0
    for p, y in zip(predicted, y_test):
        if p == 0 and y == 1:
            FN += 1
    return FN

def precision(predicted, y_test):
    TP= true_positive(predicted, y_test)
    FP= false_positive(predicted, y_test)
    return (TP)/(TP+FP)

def recall(predicted, y_test):
    TP= true_positive(predicted, y_test)
    FN= false_negative(predicted, y_test)
    return (TP)/(TP+FN)

def f1(predicted, y_test):
    test_precision= precision(predicted, y_test)
    test_recall=recall(predicted, y_test)
    return (2*(test_precision * test_recall)/(test_precision + test_recall))

def display_confusion_marix(predicted, y_test):
    TP= true_positive(predicted, y_test)
    FN= false_negative(predicted, y_test)
    TN= true_negative(predicted, y_test)
    FP= false_positive(predicted, y_test)
    header = ["\t", "Predicted Spam", "Predicted Not Spam"]
    actual_positive=["Actual Spam", TP, FN]
    actual_negative= ["Actual Not Spam", FP, TN]
    print("Confusion Matrix")
    print(f"{header[0]:<12} {header[1]:<15} {header[2]:<15}")
    print(f"{actual_positive[0]:<15} {actual_positive[1]:<15} {actual_positive[2]:<15}")
    print(f"{actual_negative[0]:<15} {actual_negative[1]:<15} {actual_negative[2]:<15}")

def main():
    X,y=read_data()
    X_train,X_test,y_train,y_test=split_data(X,y,0.7)
    predicted=train_and_predict(X_train,X_test, y_train)
    print("F1", "{:.3f}".format(f1(predicted, y_test)))
    display_confusion_marix(predicted, y_test)

main()