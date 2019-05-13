# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 15:13:15 2017

@author: christiaan.erdbrink
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

"""
Data and approach based on Hyndman and Athanasopoulos: "O Text" https://www.otexts.org/fpp/7
See also http://exponentialsmoothing.net/

Prediction intervals using bootstrap (Update Dec2018):
    https://otexts.org/fpp2/prediction-intervals.html
"""


def Holt_linear_trend(y, coeffs, h, verbose=False):
    """
    Holt's linear trend model for exponential smoothing, i.e. using level estimate and linear trend.
    h is the number of steps ahead of the forecast _after the last observation_
    """
    alpha, beta = coeffs
    level = np.empty((len(y)+1))
    level[0] = y[0]  # N.B. hard-coded l0 = y[0] !!
    trend = np.empty((len(y)+1))
    trend[0] = y[1] - y[0]  # N.B. hard-coded b0 = y2 - y1
    if verbose:
        print("\nt, y[t], level[t], trend[t], forecast[t]")
        print(0,"  -  ",level[0], trend[0]," - ")
    forecast = []
    t_forecast = []
    for t in range(1,len(y)+1):
        level[t] = alpha*y[t-1] + (1. - alpha)*(level[t-1] + trend[t-1])
        trend[t] = beta*(level[t] - level[t-1]) + (1. - beta)*trend[t-1]
        if t>1:
            forecast.append(level[t-1] + trend[t-1])
            t_forecast.append(t)
            if verbose:
                print(t_forecast[-1], y[t-1], level[t], trend[t], forecast[-1])
        else:
            if verbose:
                print(1, y[t-1], level[t], trend[t], "  -  ")
    
    # And now add extra forecast steps (beyond observations y)
    l = level[t]
    b = trend[t] 
    for h_ in range(1,h+1):
        forecast.append(l + h_*b)
        t_forecast.append(t_forecast[-1]+1) 
        if verbose:
            print(t_forecast[-1], "                                 ", forecast[-1])
    return t_forecast, forecast

def SSE(model_params, y, fun):
    """
    Sum of Squared (prediction) Error
    """
    _, y_predict = fun(y, model_params, 1)
    return np.sum((y[1:] - np.array(y_predict[:-1]))**2)  # out-of-sample forecasts versus known observations (last one not known)

def SSE_err_corr(model_params, y, fun):
    """
    Sum of Squared (prediction) Error, rewritten for Error Correction method
    """
    y_predict = fun(y, model_params)
    return np.sum((y[1:] - np.array(y_predict[:-1]))**2)  # out-of-sample forecasts versus known observations (last one not known)
    
def Holt_linear_error_correction(Y, coeffs):
    """
    Alternative formulation, should give exactly the same results.
    """
    alpha, beta = coeffs
    l = Y[0] # level
    b = Y[1] - Y[0] # trend
    err = None
    forecast = []
    for y in Y:
        err = y - (l + b)
        l += b + alpha*err
        b += alpha*beta*err
        forecast.append(l + b)
    return np.array(forecast)
    

if __name__=='__main__':
#    os.chdir('/Users/christiaan.erdbrink/Documents/Scripts/universal_tools_2017/time_series/exponential_smoothing')
    os.chdir(r'C:\Users\erdbrca\Documents\my_code\general_tools\time series')
    data = np.genfromtxt('air_passengers_data.txt', delimiter=',')
    t = data[:,0]
    y = data[:,1]
    
    # Find optimal alpha based on Sum of Squared Error
    bnds = ((0, 1), (0, 1))
    params_best = minimize(SSE, [.5, .5], args=(y, Holt_linear_trend), bounds=bnds)
    params_best = params_best.x

    # Plot data    
    plt.close('all')
    plt.figure(figsize=(12,5))
    plt.plot(t, y, 'ko-', lw=1.5)
    
    colors = 'brgymc'
    # Plot forecast with prefixed coefficients:
    t_forecast1, y_forecast1 = Holt_linear_trend(y, [.8, .2], 5)
    t_forecast1 += t[0] - 1
    plt.plot(t_forecast1, y_forecast1, 'b.-.', lw=1.5, alpha=.7, label="linear: [.8 .2]")
    
    # Plot forecast with optimised coefficients:
    t_forecast2, y_forecast2 = Holt_linear_trend(y, params_best, 8)
    t_forecast2 += t[0] - 1
    plt.plot(t_forecast2, y_forecast2, 'b:', lw=1.5, alpha=.7, label="linear: "+str(params_best))
    plt.legend()
    
    # Check whether Error-Correction method gives same answer
    y_hat_err_corr = Holt_linear_error_correction(y, [.8, .2])
    #print(abs(np.array(y_hat_err_corr) - y_forecast1[:15]).sum())
    print(np.allclose(np.array(y_hat_err_corr), y_forecast1[:15]))
    
    # Parameters optimisation using Error Correction function
    y_hat_err_corr_optimal_coeff = minimize(SSE_err_corr, [.5, .5], args=(y, Holt_linear_error_correction), bounds=bnds)
    y_hat_err_corr_optimal_coeff = y_hat_err_corr_optimal_coeff.x
    y_hat_err_corr_optimal = Holt_linear_error_correction(y, y_hat_err_corr_optimal_coeff)
#    t_ = t + 1    
#    plt.figure(figsize=(7,5))
#    plt.plot(t, y, 'ko-')
#    plt.plot(t_, y_hat_err_corr_optimal, 'g.:')

    # Plot in-sample errors histogram
    IS_Err = y[1:] - y_hat_err_corr_optimal[:-1]
#    plt.figure(); plt.hist(IS_Err, bins=9)
    
# =============================================================================
#     Compute error bounds
# =============================================================================
    df = pd.DataFrame(index=pd.date_range(start='1990', periods=len(t_forecast2), freq='y'), \
                      columns=['forecast_mean'], \
                      data=y_forecast2)
    df['hi'] = np.nan
    df['lo'] = np.nan
    # Bootstrapping from in-sample errors to create prediction intervals  [ATTEMPT]
    N = 10000
    n_steps_ahead = 6
    err_interval_lo = [df.loc['2004','forecast_mean'].iloc[0]]
    err_interval_hi = [df.loc['2004','forecast_mean'].iloc[0]]
    BootStrap = {}
    BootStrap[1] = IS_Err
    for i in range(1,n_steps_ahead):
        if i>1:
            BootStrap[i] = []
            for _ in range(N):
                BootStrap[i].append( np.random.choice(BootStrap[i-1], replace=True) )
            BootStrap[i] = np.array(BootStrap[i])
        err_interval_lo.append(err_interval_lo[-1] + np.percentile(BootStrap[i], 5))
        err_interval_hi.append(err_interval_hi[-1] + np.percentile(BootStrap[i], 95))
  
    # Plotting forecast with intervals in pandas
    t_future = pd.date_range(start='2004', periods=n_steps_ahead, freq='y')
    df.loc[t_future,'lo'] = err_interval_lo
    df.loc[t_future,'hi'] = err_interval_hi
    df.plot(figsize=(8,6))

    # HMMMMMM DOESN'T LOOK THAT GREAT!!!!!!!!  
    