import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras.layers.recurrent import LSTM
from sklearn.model_selection import train_test_split
import pandas as pd

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.embeddings import Embedding

df = pd.read_csv('E:\HvA\Big Data Scientist & Engineer\Block2\Assignment2\code_and_df\src\data.csv')

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

X = []
sentences = list(df['review'])
for sen in sentences:
    X.append(sen)


y = df.label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

max_words=5000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

maxlen = 50

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

from keras.layers.convolutional import Conv1D
from keras.layers import GlobalMaxPooling1D

vocab_size = len(tokenizer.word_index) + 1


model = Sequential()
# vocab_Size is the size of our vocabulary + 1 and it's the input_dim,
# 32 is the output_dim and its the dimension of the dense embedding,
# last thing input_length is used when you are going to connect flatten and dense layers
model.add(Embedding(vocab_size, 32, input_length=maxlen))
# text is 1 dimensional, so Conv1D,
# 128 is the number of output filters in the convolution
# the kernel in this case is a vector of length 5, not a 2 dimensional matrix,
# activation relu is the activation function max(x, 0)
model.add(Conv1D(128, 5, activation='relu'))
# the pooling layer
# in our case is 1D because we are using the model for text 
model.add(GlobalMaxPooling1D())
# Basic dense layer,
# 1 is the dimensionality of the output,
# activation is the activation function sigmoid sigmoid(x) = 1 / (1 + exp(-x))
model.add(Dense(1, activation='sigmoid'))
# loss=binary_crossentropy, it's the algorithm to calculate the losses this way:
# Computes the cross-entropy loss between true labels and predicted labels,
# optimizer=adam, is the algorithm we are going to use to improve our model
# metrics are the metrics we are going to evaluate, in this case the accuracy 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
print(model.summary())

# X_train, y_train, is the data to train the model
# batch_size is the number of samples per gradient update
# epochs are the number of trainings of the model
# verbose is for check in the terminal how is the model fitting
# validation split is a part of the training data splitted to evaluate the accuracy and losses per epoch
history = model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)
score = model.evaluate(X_test, y_test, verbose=1)

print("Loss value:", score[0]) # 0.16014418005943298
print("Test Accuracy:", score[1]) # 0.9473514556884766