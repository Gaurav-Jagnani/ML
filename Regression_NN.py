import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import Dense

X = np.random.randint(10, size=(100))
Y = X * 2 + np.random.normal(size=100)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = Sequential()
model.add(Dense(1, input_dim=1, activation="linear"))
model.compile(optimizer="sgd", loss="mean_squared_error")
model.summary()

model.fit(X_train, Y_train, epochs=10)
Y_pred = model.predict(X_test)
print("r2_score ", r2_score(Y_test, Y_pred))
print("mean_squared_error ", mean_squared_error(Y_test, Y_pred))
print("mean_absolute_error ", mean_absolute_error(Y_test, Y_pred))

