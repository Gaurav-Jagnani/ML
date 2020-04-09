import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import KFold

iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
Y = iris.target

corr_df = X.corr().abs()
up_tri = np.triu(
	np.full(corr_df.shape, 1), k=1)
up_tri = up_tri.astype(bool)
corr_df = corr_df.where(up_tri)
correlated_cols = [col for col in corr_df if any(corr_df[col] > 0.75)]

X.drop(correlated_cols, axis=1, inplace=True)

print(X.shape)
