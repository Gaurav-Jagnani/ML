import string

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN

from keras.utils import to_categorical

# "abcdefghijklmnopqrstuvwxyz"
alphabets = string.ascii_lowercase
len_alphabets = len(alphabets)
seq_len = 4
stride = 1

# Generate sequences
words = list()
labels = list()
for i in range(0, len(alphabets) - seq_len, stride):
	words.append(list(alphabets[i:i + seq_len]))
	labels.append(alphabets[i + seq_len])

# Tranform to one hot
num_sequences = len(words)
X = np.zeros((num_sequences, seq_len, len_alphabets))
Y = np.zeros((num_sequences, len_alphabets))
alphabets_indices = dict(zip(list(alphabets), range(26)))

for i in range(num_sequences):
	for j in range(seq_len):
		X[i][j][alphabets_indices[words[i][j]]] = 1
	Y[i][alphabets_indices[labels[i]]] = 1

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = Sequential()
model.add(SimpleRNN(128,
	input_shape=(seq_len, len_alphabets),
	return_sequences=True)
)
model.add(SimpleRNN(128))
model.add(Dense(len_alphabets, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

model.fit(X_train, Y_train, epochs=100)
Y_pred = model.predict(X_test)

print(np.argmax(Y_test, axis=1))
print(np.argmax(Y_pred, axis=1))
print(model.evaluate(X_test, Y_test))