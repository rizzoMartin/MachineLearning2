from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
import pandas as pd
from dask import delayed
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split
from dask_ml.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import datetime

df = dd.read_csv('E:\HvA\Big Data Scientist & Engineer\Block2\Assignment2\code_and_df\src\data.csv')

df_review = df.review.compute()

my_pattern = r'\b[^\d\W][^\d\W]+\b'
vect = TfidfVectorizer(ngram_range=(1,2), max_features=100, token_pattern=my_pattern, stop_words=ENGLISH_STOP_WORDS).fit(df_review)
X_txt = vect.fit_transform(df_review)

X = X_txt.toarray()
y = df.label.compute()

begin_time = datetime.datetime.now()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

log_reg = LogisticRegression().fit(X_train, y_train)
y_predicted = log_reg.predict(X_test)

print('time: ', datetime.datetime.now() - begin_time)
print('accuracy: ', accuracy_score(y_test, y_predicted))