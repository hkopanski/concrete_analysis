# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 08:26:35 2020

@author: hkopansk
"""

import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

plt.style.use('fivethirtyeight')

os.chdir(r"C:\Users\hkopansk\OneDrive - Biogen\Documents\Python Data")

df_cement = pd.read_csv(r'cement.csv')

df_cement['class'] = pd.cut(x = df_cement['comp_strength_Mpa'], 
                            bins = [0, 20, 40, math.inf], 
                            labels = ['Low', 'Med', 'High'])



print(df_cement.info)

cement_lm = ols('comp_strength_Mpa ~ Age_day + Cement', 
               data = df_cement).fit()

print(cement_lm.summary())

cement_anova = sm.stats.anova_lm(cement_lm, typ=2)

print(cement_anova)

fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (20, 12), 
                       dpi = 200)

sns.lineplot(x = 'Age_day', y = 'comp_strength_Mpa', 
             data = df_cement, ax = ax[0,0])

#sns.kdeplot(data = df_cement, x = 'Cement', y = 'comp_strength_Mpa', ax = ax[0,0])

sns.scatterplot(x = 'Age_day', y = 'comp_strength_Mpa', 
                data = df_cement, hue = 'Water', ax = ax[0,1])

sns.lineplot(x = 'Age_day', y = 'comp_strength_Mpa', 
             data = df_cement, ax = ax[0,1], color = 'maroon')

sns.stripplot(x = 'Superplasticizer', y = 'class', 
                data = df_cement, color = 'black', ax = ax[1,0])

sns.violinplot(data = df_cement, x = 'Superplasticizer', 
               y = 'class', linewidth = 0.5, ax = ax[1,0])

sns.scatterplot(x = 'Cement', y = 'Coarse_Aggregate', 
                data = df_cement, hue = 'comp_strength_Mpa', ax = ax[1,1])
