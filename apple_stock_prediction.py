# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 03:16:08 2019

@author: ZETH Empire
"""

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


# importing data
dataframe = pd.read_csv(filepath_or_buffer="AAPL.csv")
X = dataframe.iloc[:, [1, 2, 3, 4, 6]].values
Y = dataframe.iloc[:, 5:6].values


# scaling data
from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
scale_Y = StandardScaler()

X = scale_X.fit_transform(X)
Y = scale_Y.fit_transform(Y)



# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 1)
X = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_


"""""
# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, Y)
"""""




# Fitting Linear Regressor to the dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, Y)





# Predicting a new result
y_pred = regressor.predict(6.5)
y_pred = scale_Y.inverse_transform(y_pred)






# Visualising results
plt.scatter(X, Y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising results(for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


