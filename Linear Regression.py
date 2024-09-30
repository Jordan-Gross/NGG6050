#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 09:35:41 2024

@author: JG1898
"""

import matplotlib.pyplot as plt
import numpy as np
import random as rnd
import collections
import time
import scipy.stats as st
import pandas as pd

import scipy.stats as stats 
from IPython.display import clear_output
from operator import itemgetter
from statsmodels.stats import proportion

from numpy import matlib

age = [3,4,5,6,7,8,9,11,12,14,15,16,17]

wing_Length= [1.4,1.5,2.2,2.4,3.1,3.2,3.2,3.9,4.1,4.7,4.5,5.2,5.0]

# Create a DataFrame
RandomDataFrame = pd.DataFrame({
    "Age": age,
    "WingLength": wing_Length
})

#1 
t_statistic, p_value = stats.ttest_ind(age, wing_Length)
print("t-statistic", t_statistic)
print("p-value", p_value)

#2
plt.scatter(age,wing_Length,color = 'lightpink')           
plt.xlabel('Age')
plt.ylabel('Wing Length')
plt.show() 

#3: We can reject the null hypothesis of b=0. 

#4 Confidence Interval = x(+/-)t*(s/√n)

ci_age = st.t.interval(confidence=.95, 
              df=len(age)-1,
              loc=np.mean(age),
              scale=st.sem(age))

ci_wing_length = st.t.interval(confidence=.95, 
              df=len(wing_Length)-1,
              loc=np.mean(wing_Length),
              scale=st.sem(wing_Length))

# From Chat-GPT because I got stuck on how to plot the confidence intervals
mean_age = np.mean(age)
mean_wing_length = np.mean(wing_Length)

age_error = (ci_age[1] - ci_age[0]) / 2
wing_length_error = (ci_wing_length[1] - ci_wing_length[0]) / 2

plt.scatter(age, wing_Length, color='lightpink', label='Data points')
plt.errorbar(mean_age, mean_wing_length, xerr=age_error, yerr=wing_length_error, fmt='o', color='blue', label='95% CI')

plt.xlabel('Age')
plt.ylabel('Wing Length')
# from chat GPT a regression line 
plt.legend()
plt.show()

#5 and 6 
slope, intercept, r_value, p_value, std_err = stats.linregress(age, wing_Length)
print("Pearson's R", r_value)
r_squared = r_value**2

print("R-Squared", r_squared)

plt.scatter(age, wing_Length, color='lightpink', label='Data points')
plt.errorbar(mean_age, mean_wing_length, xerr=age_error, yerr=wing_length_error, fmt='o', color='blue', label='95% CI')

plt.xlabel('Age')
plt.ylabel('Wing Length')
# from chat GPT a regression line 
plt.plot(age, np.array(age) * slope + intercept, color='red', label='Regression Line')
plt.show()

#7 Add some noise to the data and see how the regression changes.
random = np.random.normal(1,6,size=13)
RandomDataFrame['Noise'] = random

t_statistic, p_value = stats.ttest_ind(age, RandomDataFrame['Noise'])
print("Noisy t-statistic", t_statistic)
print("Noisy p-value", p_value)

slope, intercept, r_value, p_value, std_err = stats.linregress(age, RandomDataFrame['Noise'])
print("Noisy Pearson's R", r_value)
r_squared = r_value**2

print("Noisy R-Squared", r_squared)

plt.scatter(age, wing_Length, color='lightpink')           
plt.scatter(age, RandomDataFrame['Noise'], color='lightblue')
plt.xlabel('Age')
plt.ylabel('Wing Length')
# from chat GPT a regression line 
plt.plot(age, np.array(age) * slope + intercept, color='red', label='Regression Line')
plt.show()


