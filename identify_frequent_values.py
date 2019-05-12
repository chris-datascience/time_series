# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from itertools import combinations
from random import choice
import pandas as pd
import numpy as np
import sklearn.mixture as sm
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style='dark')


#def identify_too_close(Amounts, neighbour_range=2):
#    """
#        Identify value pairs in array Amounts that are within distance neighbour_range from each other.
#    """
#    neighbour_pairs = []
#    for c in combinations(Amounts, 2):
#        if np.abs(np.diff(c))<neighbour_range:
#            neighbour_pairs.append(c)
#    return neighbour_pairs
#
#def filter_near_duplicates(amts_value_count, n_select=3, amt_vicinity=4):
#    top_amts = list(amts_value_count.index)
#    amt_pairs = identify_too_close(top_amts, amt_vicinity)  # <-- Define vicinity of equality
#    neighbours_flat = set([a for a,b in amt_pairs] + [b for a,b in amt_pairs])
#    pair_amts_to_keep = []
#    for x0,x1 in amt_pairs:
#        if top_discrete_amts.loc[x0]>top_discrete_amts.loc[x1]:
#            pair_amts_to_keep.append(x0)
#        else:
#            pair_amts_to_keep.append(x1)
#    amts_to_keep = (set(top_amts) - neighbours_flat) | set(pair_amts_to_keep)
#    final_amts = list(top_discrete_amts[top_discrete_amts.index.isin(amts_to_keep)].index[:n_select])
#    return final_amts

def discrete_selection(Amts, vc_single_threshold=.05, vc_cumul_threshold=.7, verbose=False):
    # Step [1]: Check if there are dominant discrete tran amounts in this currency
    # vc_single_threshold is proportion 0-1 of all txs, for value_count
    Amts = Amts.copy()
    Amts_vc = pd.Series(Amts).value_counts() / len(Amts)
    if verbose:
        print(Amts_vc)
    Amts_vcc = Amts_vc.cumsum()
    n_select = 0
    for vc, vcc in zip(Amts_vc,Amts_vcc):
        if all([n_select<2, vc>=3*vc_single_threshold]):
            n_select += 1
        elif all([vc>=vc_single_threshold, vcc<vc_cumul_threshold]):
            n_select += 1        
    if n_select>0:
        top_discrete_amts = set(Amts_vc.index[:n_select])
        return set(top_discrete_amts)
    else:
        return set()

def mixture_selection(Amts, n_max=3):
    # Step[2]: Run Mixture Model
    Amts = Amts.copy()
    M1 = sm.BayesianGaussianMixture(n_components=n_max, n_init=5, max_iter=1000)  # n_components is maximum components fitted
    M1.fit(Amts.reshape(-1,1))
    Weights = M1.weights_
    Covs = np.squeeze(M1.covariances_, axis=1)[:,0]
    amt_selection = (Weights>MM_weights_threshold) & (Covs<MM_var_threshold)  # <--- manually chosen identification threshold
    Means = M1.means_[amt_selection]
    Weights = M1.weights_[amt_selection]
    if len(Means)>1:
        Means = Means[np.argsort(Weights)[::-1]][:,0]  # sort acc. to descending weights (i.e. heaviest weight comes first)
    else:
        Means = Means[0]
    return Means


if __name__=='__main__':
    
    # -- Settings --
    price_duplication_filter = 3.  # range in whatever currency within which duplicate frequent transaction amounts are filtered out
    MM_weights_threshold = .1  # for mixture model
    MM_var_threshold = 10  # for mixture model
    verbose = True
    colors = 'brgmyck'*10
    
    # -- Initialise --
    Amts = np.array([1]*8 + [5]*12 + [9,0,0,14,16,22,4,2,2,3])  
    
    # -- M A I N --
    discrete_ = discrete_selection(Amts)

    # TO DO: FILTER OUT NEAR-DUPLICATES
#    Means = filter_near_duplicates(top_discrete_amts, n_select=3, amt_vicinity=price_duplication_filter)
    
    mixtures_ = mixture_selection(Amts)
    # TO DO: ADD AUTO-RULE FOR WEIGHTS AND COVARIATE THRESHOLDS (DEPENDING ON SET SIZE?)

    plt.figure(figsize=(8,6))
    plt.hist(Amts, density=True, color='b', alpha=.4)
    dens, _ = np.histogram(Amts, density=True)
    plt.vlines(list(discrete_), 0, dens.max(), colors='k', lw=3, label='valuecounts')
    plt.vlines(mixtures_, 0, dens.max(), colors='r', lw=3, label='mixmodel')
    plt.legend(fontsize=15)
