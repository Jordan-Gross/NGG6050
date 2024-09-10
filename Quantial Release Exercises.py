#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 10:27:41 2024

@author: JG1898
"""

import numpy as np
import random as rnd
import collections
import matplotlib.pyplot as plt
import time
import scipy.stats as st
import pandas as pd

from scipy.stats import bernoulli, binom, poisson, chi2
from IPython.display import clear_output
from operator import itemgetter
from statsmodels.stats import proportion

from numpy import matlib

# Exercise 1: 
p = 0.2
n = 10 
num_releases = [0,1,2,3,4,5,6,7,8,9,10]


for i in  num_releases:
    print(f"Probability of {i} releases: {binom.pmf(i, n, p)}")
    
# Exercise 2: 
    
p2 = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
quanta = 8
avaliable = 14

for i in  p2:
    print(f"Probability of 8 quanta being released with p = {i}: {binom.pmf(quanta, avaliable,i)}")

# Exercise 3:
    
p3 = 0.1
quanta2 = 5

# likelihood(p)=Probability(8 quanta released | p)× Probability (5 quanta released | p)
part1 = (binom.pmf(quanta, avaliable,p3))
part2 = (binom.pmf(quanta2, avaliable, p3))

print(f"The probability of 8 quanta being released given a p = 0.1 AND the probability of 5 quanta being released given p =0.1 is {part1*part2}")

# logL(p)=log(P(8 quanta released | p))+log(P(5 quanta released | p))
log_likelihood = np.log(part1) + np.log(part2)
print(f"The log likelihood of 8 quanta being released given a p = 0.1 AND the log likelihood of 5 quanta being released given p =0.1 is {log_likelihood}")

#likelihood functions using deciles of release probability between 0 and 1

log_likelihoods = []
for i in  p2:
    part1 = (binom.pmf(quanta, avaliable,i))
    part2 = (binom.pmf(quanta2, avaliable, i))
    likelihood = part1 * part2
    print(f"The probability of 8 quanta being released given a p = {i} AND the probability of 5 quanta being released given p ={i} is {likelihood}")
    
    # total log-likelihood functions using deciles of release probability between 0 and 1. 
    # From Chat=GPT "max(part1, 1e-10) ensures that the value passed to np.log() is never zero or negative."
    log_likelihood = np.log(max(part1, 1e-10)) + np.log(max(part2, 1e-10))
    log_likelihoods.append(log_likelihood)
    print(f"The log likelihood of 8 quanta being released given a p = {i} AND the log likelihood of 5 quanta being released given p ={i} is {log_likelihood}")

#I added a plot because I wanted to, and I was curious. 
plt.plot(p2,
         log_likelihoods,
         color = 'lightpink')           
plt.xlabel('Release Probability')
plt.ylabel('Log_likelihood of 8 and 5 quantal releases of 14 avaliable')
plt.show() 

#What is the maximum value? Based on the graph the maximum log_likelihood of release occurs at a release probability of 0.5. 
# Can you improve your estimate by computing the functions at a higher resolution? I will do the same thing again but with more possible release probabilities. 
high_log_likelihoods = []
p2_high = np.linspace(0,1.0,num=1000)
for i in  p2_high:
    part1 = (binom.pmf(quanta, avaliable,i))
    part2 = (binom.pmf(quanta2, avaliable, i))
    high_likelihood = part1 * part2
    
    # total log-likelihood functions using deciles of release probability between 0 and 1. 
    # From Chat=GPT "max(part1, 1e-10) ensures that the value passed to np.log() is never zero or negative."
    high_log_likelihood = np.log(max(part1, 1e-10)) + np.log(max(part2, 1e-10))
    high_log_likelihoods.append(high_log_likelihood)

plt.plot(p2_high,
         high_log_likelihoods,
         color = 'lightblue')           
plt.xlabel('High Resolution Release Probability')
plt.ylabel('Log_likelihood of 8 and 5 quantal releases of 14 avaliable')
plt.show() 
# How does the estimate improve as you increase the sample size? With more possible release probabilities, the graph looks more smooth with a define peak of the parabola around 0.5.

# Exercise 4: 
data_frame = pd.DataFrame({
    'Measured Releases':[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],
    'Count':[0, 0, 3, 7, 10, 19, 26, 16, 16, 5, 5, 0, 0, 0, 0]
    })

log_likelihoods_2 = []
p3 = np.arange(0,1.0,0.01)
for i in  p3:
    likelihood_2 = (binom.pmf(data_frame['Measured Releases'], avaliable,i))
    log_likelihood_2 = np.sum(data_frame['Count'] * np.log(np.maximum(likelihood_2, 1e-10)))
    log_likelihoods_2.append(log_likelihood_2)

plt.plot(p3, 
         log_likelihoods_2, 
         label='Log-Likelihood', 
         color='purple')
plt.xlabel('Release Probability')
plt.ylabel('Log_likelihood')
plt.show()
# The peak of the graph, the most likely value of p, is a release probability of approximately around 0.45. 

# Excercise 5: 

release_probability = 0.3
quantal_events = 7 

# Null hypothesis: the release probability does not change when the temperature is changed. Ie. the temperature change has no effect. 

print(binom.pmf(quantal_events, avaliable,release_probability))

# 0.06181335872712. The given p-value is not less than 0.05 which is the traditional cutoff for significane, so we fail to rekect the null hypothesis. 
# we cannot conclude that temperature has an effect. Probability of 6.18% of randomly getting this measurement if we could reject the null hypothesis. 
        
        
        
        
        
        
        