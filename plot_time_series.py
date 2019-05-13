# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#% matplotlib inline
plt.style.use('seaborn-white')


def plot_series(S, winsize=14, plot_title="", plot_ylabel="", plot_color='b'):
    """
    ma = moving average
    mstd = moving standard deviation with same windows size as ma
    See https://pandas.pydata.org/pandas-docs/stable/visualization.html  [last example]
    """
    S.fillna(0, inplace=True)
    ma = S.rolling(window=winsize, min_periods=2, win_type='triang').mean()
    mstd = S.rolling(window=winsize, min_periods=2).std()
    mstd_low = ma - 2*mstd
    mstd_low[mstd_low<0] = 0.
    mstd_hi = ma + 2*mstd

    plt.figure(figsize=(10,6))
    plt.plot(S.index, S, 'k.')
    plt.plot(ma.index, ma, plot_color+'-')
    plt.fill_between(mstd.index, mstd_low, mstd_hi, color=plot_color, alpha=0.15)
    plt.title(plot_title, fontsize=15)                                                           
    plt.ylabel(plot_ylabel, fontsize=14)
    
    
if __name__=='__main__':
    # Example:
    Y = pd.Series(data=100.+np.random.randn(200,1).cumsum(), index=pd.date_range(start='2017-01-01', periods=200, freq='D'))
    plot_series(Y, winsize=10, plot_title='Sale volumes', plot_ylabel='tx per day')