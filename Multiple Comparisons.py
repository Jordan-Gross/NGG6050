#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 10:33:14 2024

T-Test and Multiple Comparisons Exercises 

@author: JG1898
"""
# Multiple Comparisons Exercise 1 
import matplotlib.pyplot as plt
import numpy as np
import random as rnd
import collections
import time
import scipy.stats as st
import pandas as pd

from scipy.stats import bernoulli, binom, poisson, chi2
from IPython.display import clear_output
from operator import itemgetter
from statsmodels.stats import proportion

from numpy import matlib

# Tutorial 1 

alpha = 0.05
N = np.arange(0,100)
plt.plot(N, 1-(1-alpha)**N)
plt.xlabel('N')
plt.ylabel('P Error')
# As the sample size increases, the probability of an error increases exponentially. 


# First, simulate multiple (say, 1000) t-tests comparing two samples with equal means and standard deviations, and save the p-values. 
# Obviously, at p<0.05 we expect that ~5% of the simulations to yield a "statistically significant" result (of rejecting the NULL hypothesis that the samples come from distributions with equal means).

alpha = 0.05
simulations = 1000
N = 1000
mu_1 = 25
mu_2 = 25
sigma = 5
p_values = []

for i in range (simulations): 
    
    N1 = np.random.normal(mu_1, sigma, N)
    N2 = np.random.normal(mu_2, sigma, N)
    
    tstat, pval = st.ttest_ind(N1, N2)
    
    p_values.append(pval)

p_values = np.array(p_values)
significant_pvalues = []

for p in p_values: 
    if p < alpha:
        significant_pvalues.append(p)
print(significant_pvalues)
        
# Second, once you have the simulated p-values, apply both methods to address the multiple comparisons problem.

# Bonferroni Correction: divide alpha by the number of comparisons
bonferroni = alpha/simulations
bonferroni_significant_pvalues = []

for p in p_values:
    if p < bonferroni: 
        bonferroni_significant_pvalues.append(p)

print(bonferroni_significant_pvalues)
        
# Benjamini–Hochberg procedure: control the false-discovery rate 
    
# 1. Rank the individual p-values in ascending order, labeled i=1...n
# From google because I was not sure how to do the ranking part of the question. 
ranked_list = sorted(enumerate(p_values), key=lambda x: x[1])

# 2. For each p-value, calculate its "critical value" as (i/n)Q, where i is the rank, n is the total number of tests, and Q is the false discovery rate (a percentage) that you choose (typically 0.05).
q= 0.05
critical_values =[]

for i in p_values: 
    critical_value = [(i+1)/ N*q]
    critical_values.append(critical_value)

# 3. In your rank-ordered, original p-values, find the largest value that is smaller than its associated critical value; this p-value is the new criterion (i.e., reject for all cases for which p ≤ this value).

# I got stuck here and used Chat-GPT: 

# Find the largest p-value that is smaller than its associated critical value
new_criterion = 0

for i in range(N):
    if ranked_list[i][1] <= critical_values[i]:
        new_criterion = ranked_list[i][1]

# Identify significant p-values using the new criterion
BH_significant_p_values = [(idx, p_val) 
                        for idx, p_val in enumerate(p_values) 
                        if p_val <= new_criterion]
print(f"Number of significant results: {len(BH_significant_p_values)}")
print("Significant p-values:")

for idx, p_val in BH_significant_p_values:
    print(f"Original Index: {idx}, p-value: {p_val}")

#--------------------------------------------------------------------------#
# Third, set the sample 1 and sample 2 means to be 1 and 2 respectively, and re-run the exercise. 
alpha = 0.05
simulations = 1000
N = 1000
mu_1 = 1
mu_2 = 2
sigma = 5
p_values = []

for i in range (simulations): 
    
    N1 = np.random.normal(mu_1, sigma, N)
    N2 = np.random.normal(mu_2, sigma, N)
    
    tstat, pval = st.ttest_ind(N1, N2)
    
    p_values.append(pval)

p_values = np.array(p_values)
significant_pvalues = []

for p in p_values: 
    if p < alpha:
        significant_pvalues.append(p)
print(significant_pvalues)
        
# Second, once you have the simulated p-values, apply both methods to address the multiple comparisons problem.

# Bonferroni Correction: divide alpha by the number of comparisons
bonferroni = alpha/simulations
bonferroni_significant_pvalues = []

for p in p_values:
    if p < bonferroni: 
        bonferroni_significant_pvalues.append(p)

print(bonferroni_significant_pvalues)
        
# Benjamini–Hochberg procedure: control the false-discovery rate 
    
# 1. Rank the individual p-values in ascending order, labeled i=1...n
ranked_list = sorted(enumerate(p_values), key=lambda x: x[1])

# 2. For each p-value, calculate its "critical value" as (i/n)Q, where i is the rank, n is the total number of tests, and Q is the false discovery rate (a percentage) that you choose (typically 0.05).
q= 0.05
critical_values =[]

for i in p_values: 
    critical_value = [(i+1)/ N*q]
    critical_values.append(critical_value)

# 3. In your rank-ordered, original p-values, find the largest value that is smaller than its associated critical value; this p-value is the new criterion (i.e., reject for all cases for which p ≤ this value).

# I got stuck here and used Chat-GPT: 

# Find the largest p-value that is smaller than its associated critical value
new_criterion = 0

for i in range(N):
    if ranked_list[i][1] <= critical_values[i]:
        new_criterion = ranked_list[i][1]

# Identify significant p-values using the new criterion
BH_significant_p_values = [(idx, p_val) 
                        for idx, p_val in enumerate(p_values) 
                        if p_val <= new_criterion]
print(f"Number of significant results: {len(BH_significant_p_values)}")
print("Significant p-values:")

for idx, p_val in BH_significant_p_values:
    print(f"Original Index: {idx}, p-value: {p_val}")

# What do you notice? 
# With different sample means, there are far more significant p-values. 

#--------------------------------------------------------------------------#

#What if you make the difference between means even greater?
# An even larger difference in means made all of the p_values significant regardelss of correction method. 

alpha = 0.05
simulations = 1000
N = 1000
mu_1 = 2
mu_2 = 8
sigma = 5
p_values = []

for i in range (simulations): 
    
    N1 = np.random.normal(mu_1, sigma, N)
    N2 = np.random.normal(mu_2, sigma, N)
    
    tstat, pval = st.ttest_ind(N1, N2)
    
    p_values.append(pval)

p_values = np.array(p_values)
significant_pvalues = []

for p in p_values: 
    if p < alpha:
        significant_pvalues.append(p)
print(significant_pvalues)
        
# Second, once you have the simulated p-values, apply both methods to address the multiple comparisons problem.

# Bonferroni Correction: divide alpha by the number of comparisons
bonferroni = alpha/simulations
bonferroni_significant_pvalues = []

for p in p_values:
    if p < bonferroni: 
        bonferroni_significant_pvalues.append(p)

print(bonferroni_significant_pvalues)
        
# Benjamini–Hochberg procedure: control the false-discovery rate 
    
# 1. Rank the individual p-values in ascending order, labeled i=1...n
ranked_list = sorted(enumerate(p_values), key=lambda x: x[1])

# 2. For each p-value, calculate its "critical value" as (i/n)Q, where i is the rank, n is the total number of tests, and Q is the false discovery rate (a percentage) that you choose (typically 0.05).
q= 0.05
critical_values =[]

for i in p_values: 
    critical_value = [(i+1)/ N*q]
    critical_values.append(critical_value)

# 3. In your rank-ordered, original p-values, find the largest value that is smaller than its associated critical value; this p-value is the new criterion (i.e., reject for all cases for which p ≤ this value).

# I got stuck here and used Chat-GPT: 

# Find the largest p-value that is smaller than its associated critical value
new_criterion = 0

for i in range(N):
    if ranked_list[i][1] <= critical_values[i]:
        new_criterion = ranked_list[i][1]

# Identify significant p-values using the new criterion
BH_significant_p_values = [(idx, p_val) 
                        for idx, p_val in enumerate(p_values) 
                        if p_val <= new_criterion]
print(f"Number of significant results: {len(BH_significant_p_values)}")
print("Significant p-values:")

for idx, p_val in BH_significant_p_values:
    print(f"Original Index: {idx}, p-value: {p_val}")



