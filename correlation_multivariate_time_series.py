# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 10:01:35 2017

@author: Kris
"""
import os
import collections
import numpy as np


def cross_covariance_matrix(X, lag=0):
    """ X is data matrix with time series.
        Its size is T x k, with T the series length and k the number of series (vector dim).
        So first series is X[:,0], etc.
    """
    
    # --Manually: [CHECK, A BIT OFF!]--
    means = X.mean(axis=0)
    if lag>0:        
        X_lag = np.roll(X.copy(), -lag, axis=0)
        X_lag[:lag,:] = np.nan
    else:
        X_lag = X.copy()

#    gamma = np.dot((X[lag:,:] - X[lag:,:].mean(axis=0)).T, (X[:len(X)-lag,:] - X[:len(X)-lag,:].mean(axis=0))) / float(len(X)-1)
    gamma = np.dot((X[lag:,:] - means).T, (X[:len(X)-lag,:] - means)) / float(len(X)-1)
#    gamma = np.cov(X[lag:,0], X[:len(X)-lag,1], rowvar=False) # Hmm not the same..
    return gamma

def cross_correlation_matrix(gamma, X):
    k = X.shape[1]
    D = X.std(axis=0) * np.eye((k))
    D_inv = np.linalg.inv(D)
    rho = np.dot(np.dot(D_inv, gamma), D_inv)
    return rho

def compute_corr_matrices_lags(X, verbose=True):
    n_lags = 5
    k = X.shape[1]  # number of time series
    Gamma = np.zeros((n_lags+1, k, k))
    Rho = np.zeros((n_lags+1, k, k))
    for lag in range(n_lags+1):
        Gamma[lag,:,:] = cross_covariance_matrix(X, lag)
        Rho[lag,:,:] = cross_correlation_matrix(Gamma[lag,:,:], X)
        if verbose:
            print('\nlag = %i:'%lag)
            print(Rho[lag,:,:])

    # N.B. Should be the same as np.cov():
    #print np.abs(Gamma[0,:,:] - np.cov(X, rowvar=False))
    return Rho

def get_data(example):
    data = np.genfromtxt(example.filename)
    if example.section=='8.1':
        return data[1:,1:]
    else:
        return data[1:,:]


if __name__=='__main__':    
    example_case = collections.namedtuple('Example',['section', 'filename'])
    example1 = example_case('8.1','m-ibmsp2608.txt')
    example2 = example_case('8.2','m-bnd.txt')
    
    corr_matrices_lags = compute_corr_matrices_lags(get_data(example2)) # <-- choose example
    
    
    
    