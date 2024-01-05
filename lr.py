from pandas import DataFrame
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
import warnings
warnings.filterwarnings("ignore")
import datetime

df = pd.read_csv('E:\HvA\Big Data Scientist & Engineer\Block2\Assignment2\code_and_df\src\data_okay.csv')

my_pattern = r'\b[^\d\W][^\d\W]+\b'
vect = TfidfVectorizer(ngram_range=(1,2), max_features=100, token_pattern=my_pattern, stop_words=ENGLISH_STOP_WORDS).fit(df.review)
X_txt = vect.transform(df.review)

X = pd.DataFrame(X_txt.toarray(), columns=vect.get_feature_names())
y = df.label

begin_time = datetime.datetime.now()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)

log_reg = LogisticRegression().fit(X_train, y_train)
y_predicted = log_reg.predict(X_test)

print('time: ', datetime.datetime.now() - begin_time)
print('accuracy: ', accuracy_score(y_test, y_predicted))
