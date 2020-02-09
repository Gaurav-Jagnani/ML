""" Basic CNN using MNIST
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# See some images
for i in range(25):
	plt.subplot(5, 5, i+1)
	plt.xlabel(Y_train[i])
	plt.imshow(X_train[i])

plt.show()

Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)
X_train = X_train.reshape((-1, 28, 28, 1))
X_test = X_test.reshape((-1, 28, 28, 1))

# Build model
model = Sequential()
model.add(Conv2D(32, (3,3), activation="relu", input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(8, (3,3), activation="relu"))
model.add(Flatten())
model.add(Dense(32, activation="relu"))
model.add(Dense(10, activation="sigmoid"))

model.summary()

model.compile(
	optimizer="adam",
	loss="categorical_crossentropy",
	metrics=["accuracy"]
)

model.fit(X_train, Y_train, epochs=10, batch_size=128)

Y_pred = model.predict(X_test)
print(model.evaluate(X_test, Y_test))

Y_pred = np.argmax(Y_pred, axis=1)
Y_test = np.argmax(Y_test, axis=1)

print("accuracy_score: ", accuracy_score(Y_test, Y_pred))
