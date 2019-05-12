# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 17:22:26 2018

@author: Kris
"""

"""
    Code snippets for Exploratory Data Analysis, feature engineering and feature selection.
    Numerous sources, e.g. 
        https://towardsdatascience.com/bayesian-linear-regression-in-python-using-machine-learning-to-predict-student-grades-part-1-7d0ad817fca5
        Fraud Analytics textbook
        paper on RIDIT
        ...etc.
    Visualization sources:
        https://towardsdatascience.com/visualizing-data-with-pair-plots-in-python-f228cf529166
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df - pd.DataFrame() # 'Grade' is target var/


# --- INSPECT DATA IN DATAFRAME ---

# Make one plot for each different location
sns.kdeplot(df.ix[df['address'] == 'U', 'Grade'], 
            label = 'Urban', shade = True)
sns.kdeplot(df.ix[df['address'] == 'R', 'Grade'], 
            label = 'Rural', shade = True)
plt.xlabel('Grade')
plt.ylabel('Density')
plt.title('Density Plot of Final Grades by Location')

# Create the default pairplot of all variables in df
sns.pairplot(df)
sns.pairplot(df, hue = 'columnX') # using certain variable for colors



# --- FEATURE SELECTION ---
# (1) For continuous variables
df.corr()['Grade'].sort_values(ascending=False)  # from here, simply select top 5 (and most negative), e.g.

# (2) For categorical variables
# Select only categorical variables
category_df = df.select_dtypes('object')
dummy_df = pd.get_dummies(category_df)  # One hot encode the variables
dummy_df['Grade'] = df['Grade']  # Put the grade back in the dataframe
dummy_df.corr()['Grade'].sort_values(ascending=False)  # Find correlations with grade

