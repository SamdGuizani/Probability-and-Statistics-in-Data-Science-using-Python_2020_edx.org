# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 14:34:21 2020

@author: sagu-adm
"""

#%% Import modules
import pandas as pd
import numpy as np
from scipy.stats import norm, t


#%% Functions
def T_test(n,sample_mean, mu, S, test_type):
    '''1-sample T-Test to compare sample mean to a theoretical mean mu.
    Prints sample mean, sample t-score and p-value under H0.
    
    Inputs:
        n = number of observations in the sample
        sample_mean = observed sample mean
        mu = hypothetized population mean
        S = sample standard deviation
        test_type = string to define alternative hypothesis:
            "mean < μ under null hypothesis"
            "mean > μ under null hypothesis"
            "mean ≠ μ under null hypothesis"'''
            
    print("Sample mean:{:.4f}".format(sample_mean))
    t_score = (sample_mean - mu)*np.sqrt(n)/S
    print("t-score:{:.4f}".format(t_score))
    if test_type=="mean > μ under null hypothesis":
        p = 1 - t.cdf(t_score,n-1)
        print("p-value: {:.6f}".format(p))
    elif test_type=="mean < μ under null hypothesis":
        p = t.cdf(t_score,n-1)
        print("p-value : {}".format(p))
    elif test_type=="mean ≠ μ under null hypothesis":
        p = 2*(1-t.cdf(np.abs(t_score,n-1)))
        print("p-value: {}".format(p))
        
def Z_test(n,sample_mean, mu, sigma, test_type):
    '''1-sample Z-Test to compare sample mean to a theoretical mean mu.
    Prints sample mean, sample z-score and p-value under H0.
    
    Inputs:
        n = number of observations in the sample
        sample_mean = observed sample mean
        mu = hypothetized population mean
        sigma = KNOWN population standard deviation
        test_type = string to define alternative hypothesis:
            "mean < μ under null hypothesis"
            "mean > μ under null hypothesis"
            "mean ≠ μ under null hypothesis"'''
            
    print("Sample mean:{:.4f}".format(sample_mean))
    z = (sample_mean - mu)*np.sqrt(n)/sigma
    print("z-score:{:.4f}".format(z))
    if test_type=="mean > μ under null hypothesis":
        p = 1 - norm.cdf(z)
        print("p-value: {:.6f}".format(p))
    elif test_type=="mean < μ under null hypothesis":
        p = norm.cdf(z)
        print("p-value : {}".format(p))
    elif test_type=="mean ≠ μ under null hypothesis":
        p = 2*(1-norm.cdf(np.abs(z)))
        print("p-value: {}".format(p))


#%% Main Script

df = pd.read_csv('temperature.csv')

SanDiego = df[['datetime', 'San Diego']]
# SanDiego.loc[:, ['datetime']] = pd.to_datetime(SanDiego['datetime'])
pd.to_datetime(SanDiego['datetime'])

# Convert Temperature from K to °C
SanDiego.loc[:, ['San Diego']] = SanDiego['San Diego'] - 273.15

# Extract year 2016 temperature (°C) data 
Extract = SanDiego[(SanDiego['datetime'] >= '2016-01-01 00:00:00') 
                   & (SanDiego['datetime'] < '2017-01-01 00:00:00')]

sample_mean = Extract['San Diego'].mean()
S = Extract['San Diego'].var(ddof=1)**0.5
n = Extract.shape[0]

#%% t-test

# H0
test_type = 'mean < μ under null hypothesis'
# test_type = 'mean > μ under null hypothesis'
# test_type = 'mean ≠ μ under null hypothesis'
mu = 18 # hypothetized mean temperature

T_test(n,sample_mean, mu, S, test_type)

#%% Z-test

# H0
test_type = 'mean < μ under null hypothesis'
# test_type = 'mean > μ under null hypothesis'
# test_type = 'mean ≠ μ under null hypothesis'
mu = 18 # hypothetized mean temperature

Z_test(n,sample_mean, mu, S, test_type) # In this case sigma is assumed to be known and equal to measured std dev


#%% Solution:

import numpy as np
import pandas as pd
from scipy.stats import t

df = pd.read_csv('./temperature.csv')
df = df[ df['datetime'].str.contains('2016')]
df = df.loc[:, ['San Diego']]

temperature = df.values
temperature = temperature[~np.isnan(temperature)]

n = len(temperature)
mu = np.mean(temperature)
s = np.std(temperature, ddof=1)

p = t.cdf((mu - 273.15 - 18) / (s / (n ** 0.5)), n - 1)

print('p-value is {}'.format(p))


