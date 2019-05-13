# -*- coding: utf-8 -*-
"""
Created on Thu May 24 14:08:24 2018

@author: erdbrca
"""

"""
Implementation of EMA for running mean and st.dev. for floating points
name: Welford's method
Sources:
    https://dsp.stackexchange.com/questions/811/determining-the-mean-and-standard-deviation-in-real-time#892
    https://www.embeddedrelated.com/showarticle/785.php
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
"""

import numpy as np
import matplotlib.pyplot as plt


def EMA_regular(x_array, alpha=.4):
    """
    alpha needs to be 0 < alpha < 1
    """
    #T, tau = .5, .5
    #alpha = 1 - np.exp (2*np.pi*T/tau)        
    mean = x_array[0]
    Mean = [mean]
    meansq = mean**2
    SDV = [1]
    for x in x_array[1:]:
        # update the estimate of the mean and the mean square:
        mean = (1 - alpha)*mean + alpha*x
        meansq = (1 - alpha)*meansq + alpha*(x**2)
#        print(meansq)
        
        # calculate the estimate of the variance:
        var = meansq - mean**2
        std = np.sqrt(var) # standard deviation

        Mean.append(mean)
        SDV.append(std)
    return Mean, SDV

def EMA_irregular(t_array, x_array, alpha=3.):
    """
    t_array are the irregular time
    x_array are the values
    
    See: https://oroboro.com/irregular-ema/
    
    Note:
        here alpha CAN be > 1   (?!)
    """
    time = t_array[0]
    mean = x_array[0]
    Mean = [mean]
    meansq = mean**2
    SDV = [1]
    x_old = x_array[0]
    for t,x in zip(t_array[1:], x_array[1:]):
        delta_time = t - time
        time = t
        a = delta_time / alpha
        u = np.exp(-a)
        v = (1 - u) / a
        mean = u*mean + (v - u)*x_old + (1 - v)*x

        a_ = delta_time / alpha
        u_ = np.exp(-a_)
        v_ = (1 - u_) / a_
        meansq = u_*meansq + (v_ - u_)*(x_old**2) + (1 - v_)*(x**2)
        var = meansq - mean**2 # variance estimate 
        std = np.sqrt(var) # standard deviation

        x_old = x

        Mean.append(mean)
        SDV.append(std)
    return Mean, SDV

x2 = 1000 + 2.*np.random.randn(40).cumsum()
#x2 = 50 + .5*np.linspace(0,100,101) + 2.6*np.random.randn(101)    
t_regular = np.arange(len(x2))
t_irregular = np.random.randint(1, 5, size=40).cumsum()    

mm, ss = EMA_regular(x2)
plt.figure(figsize=(10,6))
plt.plot(x2, 'b.-', label='observations')
plt.fill_between(t_regular, np.array(mm) + np.array(ss), np.array(mm) - np.array(ss), color='r', alpha=.3)
plt.plot(mm, 'r-', label='EMA')
plt.title('EMA on regular time grid', fontsize=15)
plt.legend(fontsize=14)


m, s = EMA_irregular(t_irregular, x2)
plt.figure(figsize=(10,6))
plt.plot(t_irregular, x2, 'b.-', label='observations')
plt.fill_between(t_irregular, np.array(m) + np.array(s), np.array(m) - np.array(s), color='g', alpha=.3)
plt.plot(t_irregular, m, 'g-', label='irr. EMA')
plt.title('EMA on irregular time grid', fontsize=15)
plt.legend(fontsize=14)
