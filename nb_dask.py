import sys
import csv
from typing import List
import time
from joblib import parallel
import pandas as pd
from pandas.core.series import Series
import numpy as np
import re
import string
#nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
# Import the required vectorizer package and stop words list
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB    
#DASK
from dask import delayed
import os
import dask.dataframe as dd
from dask_glm.datasets import make_regression
from dask_ml.model_selection import train_test_split
from dask_ml.naive_bayes import GaussianNB
from dask_glm.datasets import make_classification
from dask_ml.linear_model import LogisticRegression
from dask_ml.wrappers import Incremental
from sklearn.metrics import accuracy_score
import datetime

df = dd.read_csv('E:\HvA\Big Data Scientist & Engineer\Block2\Assignment2\code_and_df\src\data.csv')

df_review = df.review.compute()

my_pattern = r'\b[^\d\W][^\d\W]+\b'
vect = TfidfVectorizer(ngram_range=(1,2), max_features=100, token_pattern=my_pattern, stop_words=ENGLISH_STOP_WORDS).fit(df_review)
X_txt = vect.fit_transform(df_review)

X = X_txt.toarray()
y = df.label.compute()

print('start')
begin_time = datetime.datetime.now()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

bayes = GaussianNB().fit(X_train, y_train)
y_predicted = bayes.predict(X_test)

print('time: ', datetime.datetime.now() - begin_time)
print('accuracy: ', accuracy_score(y_test, y_predicted.compute()))