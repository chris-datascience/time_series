# -*- coding: utf-8 -*-
"""
source:
Guiseppe's Glowing Python blog.

http://glowingpython.blogspot.co.uk/search?updated-max=2015-10-13T11:03:00%2B01:00&max-results=3
https://glowingpython.blogspot.com/2015/
"""

import os
import pandas as pd
import numpy as np

#os.chdir(r'C:\Users\erdbrca\Documents\GP_forecast\tempdata')
df = pd.read_csv('NZAlcoholConsumption.csv')
to_forecast = df.TotalBeer.values
dates = df.DATE.values


# N.B. Trick to reorganise time series in a matrix (in order to train on windowed periods)
def organize_data(to_forecast, window, horizon):
    """
     Input:
      to_forecast, univariate time series organized as numpy array
      window, number of items to use in the forecast window
      horizon, horizon of the forecast
     Output:
      X, a matrix where each row contains a forecast window
      y, the target values for each row of X
    """
    shape = to_forecast.shape[:-1] + \
            (to_forecast.shape[-1] - window + 1, window)
    strides = to_forecast.strides + (to_forecast.strides[-1],)
    X = np.lib.stride_tricks.as_strided(to_forecast, 
                                        shape=shape, 
                                        strides=strides)
    y = np.array([X[i+horizon][-1] for i in range(len(X)-horizon)])
    return X[:-horizon], y

k = 4   # number of previous observations to use
h = 1   # forecast horizon
X,y = organize_data(to_forecast, k, h)
# Now, X is a matrix where the i-th row contains the lagged variables xn−k,...,xn−2,xn−1 and y[i] contains the i-th target value. 


# train model:
from sklearn.linear_model import LinearRegression
 
m = 10 # number of samples to take in account
regressor = LinearRegression(normalize=True)
regressor.fit(X[:m], y[:m])

"""We trained our model using the first 10 observations, which means that we used the data from 1st quarter of 2000 to the 2nd quarter of 2002. 
We use a lag order of one year and a forecast horizon of 1 quarter. 
To estimate the error of the model we will use the mean absolute percentage error (MAPE). 
Computing this metric to compare the forecast of the remaining observation of the time series and the actual observations we have:
"""
def mape(ypred, ytrue):
    """ returns the mean absolute percentage error """
    idx = ytrue != 0.0
    return 100*np.mean(np.abs(ypred[idx]-ytrue[idx])/ytrue[idx])

print('The error is %0.2f%%' % mape(regressor.predict(X[m:]),y[m:]))

# Now plotting:
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.plot(y, label='True demand', color='#377EB8', linewidth=2)
plt.plot(regressor.predict(X), 
     '--', color='#EB3737', linewidth=3, label='Prediction')
plt.plot(y[:m], label='Train data', color='#3700B8', linewidth=2)
plt.xticks(np.arange(len(dates))[1::4],dates[1::4], rotation=45)
plt.legend(loc='upper right')
plt.ylabel('beer consumed (millions of litres)')
plt.show()