# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 22:19:31 2017

@author: Kris
"""

import os
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from correlation_multivariate_time_series import cross_covariance_matrix, cross_correlation_matrix
from sklearn.metrics import r2_score

def compute_GMVP(Cov_returns):
    """
    Computing the Global minimum variance portfolio for a certain (fitted or unfitted) set of series of assets returns.
    See pg. 473 of Tsay, section 9.2.1
    """
    pass
    
    
def single_factor_model(X):
    """
    This is the model as described in Fin.Time Series textbook by Tsay.
    Section 9.2.1
    """
    SP5 = X[:,-1].reshape(-1,1)
    assets = X[:,:-1]
    
    # Inspect assets series
    #print assets.shape
    assets_df = pd.DataFrame(data=assets)
    print('\nCorrelation matrix of original assets:')
    print(assets_df.corr().round(1))
    
    # Performing(training) a Multivariate Linear Regression model
    mlm = LinearRegression(fit_intercept=True)
#    mlm.fit(independent_vars, assets)
    mlm.fit(SP5, assets)
    mlm.get_params()

    alphas = mlm.intercept_.reshape(-1,1)
    betas = mlm.coef_
    print('\nBeta_hat:'); print(betas)
    AllCoeffs = np.hstack((betas, alphas)).T

    independent_vars = np.hstack((SP5, np.ones((len(SP5),1))))  
    MarketModel_series = np.dot(independent_vars, AllCoeffs)  # matrix G times ksi hat in section 9.2
    Residuals = assets - MarketModel_series  # matrix E in section 9.2
    
    print('\nVerification:')
    print(mlm.predict(SP5) == MarketModel_series)   # ..So using that would have been easier!
    
    print('\nR squared:')
    R_squared = np.empty((assets.shape[1],))
    for i in xrange(assets.shape[1]):
        R_squared[i] = r2_score(assets[:,i], MarketModel_series[:,i])
    print(R_squared.round(3))
    
    D = np.dot(Residuals.T, Residuals) / (float(len(SP5)) - 2)
    sigmas = np.sqrt(np.diag(D))
    print('\nSigmas:'); print(sigmas)

    # Inspect residuals:
    print('\nCorrelations residuals:')
    Res_df = pd.DataFrame(data=Residuals)
    #print Residuals.shape
    print(Res_df.corr().round(2))

    # Inspect returns under market model (i.e. fitted series):
    print('\nCorrelations under market model:')
    factorseries = pd.DataFrame(data=SP5)
    Cov_factors = factorseries.cov()  # Equals unity for a single factor!
    Cov_model = np.dot(betas, np.dot(np.array(Cov_factors), betas.T)) + D # Equation under (9.2): Cov = beta*Cov(f)*beta.T + S
    Corr_model = Cov_model / np.outer(sigmas, sigmas)
    print(Corr_model.round(3))  # INCORRECT!
#    Model_series_plus_specific_factors = pd.DataFrame(data=MarketModel_series)
#    Corr_model = Model_series_plus_specific_factors.corr()
        
    # PLOTS:
    # betas:
    asset_names = ['asset '+str(i) for i in xrange(betas.shape[0])]
    df2 = pd.DataFrame(data=betas, index=asset_names, columns=['beta_1'])
    df2.index.name='Asset'
    df2.plot.bar(figsize=(10,3))
    plt.title('Betas per asset', fontsize=15)
    
    # R^2:
    df3 = pd.DataFrame(data=R_squared, index=asset_names, columns=['R^2'])
    df3.index.name='R_squared'
    df3.plot.bar(figsize=(10,3), color='DarkGreen')
    plt.title('R squared values per asset', fontsize=15)
    

def BARRA_factor_model(X2):
    # Hot-encoded industry factors (which act as factor Betas)
    industry_hot_factors = np.zeros((10,3))
    industry_hot_factors[:4,0] = 1
    industry_hot_factors[4:7,1] = 1
    industry_hot_factors[7:,2] = 1

    # Compute correlation matrix
    asset_returns = X2 - X2.mean(axis=0)
    Cov_matrix = cross_covariance_matrix(asset_returns)
    Corr_matrix = cross_correlation_matrix(Cov_matrix, asset_returns)
    
    # Get initial estimate of factors using OLS:
    ols = LinearRegression(fit_intercept=False, n_jobs=1)
    ols.fit(industry_hot_factors, asset_returns.T)
#    ols.coef_.shape
    ols_residuals = asset_returns - np.dot(ols.coef_, industry_hot_factors.T)
    # Residual Variances captured in diagonal matrix:
    W0 = np.sqrt(np.var(ols_residuals, axis=0))
    D0 = np.diag(W0)
    
    # Now Generalized Least Squares estimate:
    gls = LinearRegression(fit_intercept=False, n_jobs=1)
    gls.fit(industry_hot_factors, asset_returns.T, sample_weight=W0)
    gls_residuals = asset_returns - np.dot(gls.coef_, industry_hot_factors.T)
    Wg = np.sqrt(np.var(gls_residuals, axis=0))
    Dg = np.diag(Wg)
    
    # Covariance of estimated factor realization:
    Cov_factors = cross_covariance_matrix(gls.coef_)
    # Covariance of excess returns:
    Cov_excess_returns = np.dot(np.dot(industry_hot_factors, Cov_factors), industry_hot_factors.T) + Dg
#    q = np.sqrt(np.diag(Cov_excess_returns))
#    Cross_corr_excess_returns = Cov_excess_returns / np.outer(q,q)
#    Cross_corr_excess_returns = cross_correlation_matrix(Cov_excess_returns, gls.coef_)
    return

def multi_factor_model(Asset_series, Factor_series):
    """
    This is simply the same implementation as used for the single factor model!
    See again Sections 9.1, 9.2 and 9.2.1 of Tsay's textbook.
    """
    # Inspect assets series
#    assets_df = pd.DataFrame(data=Asset_series)
#    print('\nCorrelation matrix of original assets:')
#    print assets_df.corr().round(1)
    
    # Performing(training) a Multivariate Linear Regression model
    mlm = LinearRegression(fit_intercept=True)
    mlm.fit(Factor_series, Asset_series)
    mlm.get_params()

#    alphas = mlm.intercept_.reshape(-1,1)
    betas = mlm.coef_
    print('\nBetas:'); print(betas)  # i.e. beta hat (now a matrix instead of a vector)
#    AllCoeffs = np.hstack((betas, alphas)).T
#    independent_vars = np.hstack((SP5, np.ones((len(SP5),1))))  
    MarketModel_series = mlm.predict(Factor_series)  # matrix G times ksi hat in section 9.2
    Residuals = Asset_series - MarketModel_series  # matrix E in section 9.2
    
    print('\nR squared:')
    R_squared = np.empty((Asset_series.shape[1],))
    for i in xrange(Asset_series.shape[1]):
        R_squared[i] = r2_score(Asset_series[:,i], MarketModel_series[:,i])
    print(R_squared.round(2))
    
    # Inspect residuals:
    print('\nCorrelations residuals:')
    Res_df = pd.DataFrame(data=Residuals)
    #print Residuals.shape
    print(Res_df.corr().round(2))
    
    # PLOTS:
    # betas:
    asset_names = ['asset '+str(i) for i in xrange(betas.shape[0])]
    df2 = pd.DataFrame(data=betas, index=asset_names, columns=['beta_1'])
    df2.index.name='Asset'
    df2.plot.bar(figsize=(10,3))
    plt.title('Betas per asset', fontsize=15)
    
    # R^2:
    df3 = pd.DataFrame(data=R_squared, index=asset_names, columns=['R^2'])
    df3.index.name='R_squared'
    df3.plot.bar(figsize=(10,3), color='DarkGreen')
    plt.title('R squared values per asset', fontsize=15)
    return

def get_data(example):
    data = np.genfromtxt(example.filename)
    if example.section=='8.1':
        return data[1:,1:]
    elif example.section=='8.2':
        return data[1:,:]
    elif example.section=='9.2.1':
        print('\n---Single Factor Model---\n')
        return data[1:,:]
    elif example.section=='9.3.1':
        print('\n\n---BARRA Factor Model---\n')
        return data[1:,:]


if __name__=='__main__':    
    example_case = collections.namedtuple('Example',['section', 'filename', 'case_name'])
    
    # Section 9.2.1: Single factor model
    example1 = example_case('9.2.1','m-fac9003.txt', 'single factor model')
    X = get_data(example1)
    single_factor_model(X)
    
#    # Section 9.3.1: BARRA fundamental factor model
#    example2 = example_case('9.3.1','m-barra-9003.txt', 'Barra factor model')
#    X2 = get_data(example2)
#    BARRA_factor_model(X2)
#
#    # Section 9.X: Multi-factor model
#    del X
#    example3 = example_case('9.XX', '', 'multi-factor model')
#    X = get_data(example3)
    