import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

x = np.random.uniform(1, 10, size=100)
y = x * 2 + np.random.normal(1, 2.5, size=100)
plt.scatter(y, x)
plt.show()

df = pd.DataFrame({"X": x, "Y": y})

scaler = StandardScaler()
df[["X", "Y"]] = scaler.fit_transform(df)

r2_scores = list()
mean_errors = list()

kfold = KFold(n_splits=5)
for train_indices, test_indices in kfold.split(df):
	X_train = df["X"][train_indices]
	Y_train = df["Y"][train_indices]
	X_test = df["X"][test_indices]
	Y_test = df["Y"][test_indices]
	X_train = pd.DataFrame(X_train)
	X_test = pd.DataFrame(X_test)

	model = LinearRegression()
	model.fit(X_train, Y_train)
	Y_pred = model.predict(X_test)

	r2_scores.append(r2_score(Y_test, Y_pred))
	print("mean_squared_error: ", mean_squared_error(Y_test, Y_pred))
	print("mean_absolute_error: ", mean_absolute_error(Y_test, Y_pred))
	print("model.coef_: ", model.coef_)
	print("model.intercept_: ", model.intercept_)
	mean_errors.append(mean_absolute_error(Y_test, Y_pred))

print("Mean r2 ", np.mean(r2_scores))
print("Mean error ", np.mean(mean_errors))