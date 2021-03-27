#!/usr/bin/env python3
# Reference: https://www.kaggle.com/residentmario/variance-inflation-factors-with-nyc-building-sales
# Reference: https://etav.github.io/python/vif_factor_python.html
import pandas as pd
import numpy as np
from patsy import dmatrices
from statsmodels.regression.linear_model import OLS


df = pd.read_csv('loan_small.csv')
df.dropna()
df = df._get_numeric_data() #drop non-numeric cols

feature_cols = ['annual_inc', 'int_rate', 'emp_length', 'dti', 'delinq_2yrs', 'revol_util', 'total_acc', 'bad_loan', 'longest_credit_length']
target = 'loan_amnt'
all_cols = feature_cols + [target]
df = df[all_cols].dropna() #subset the dataframe

print(df.head())

features = "+".join(feature_cols)

# Use the dmatrices method to construct the feature data frame (Panda). 
_, X = dmatrices(target + '~' + features, df, return_type='dataframe')

def VIF(exogenous, exogenous_idx):
    k_vars = exogenous.shape[1]
    x_i = exogenous[:, exogenous_idx]
    mask = np.arange(k_vars) != exogenous_idx
    x_noti = exogenous[:, mask]
    r_squared_i = OLS(x_i, x_noti).fit().rsquared
    vif = 1. / (1. - r_squared_i)
    return vif

# For each X, calculate VIF and save in dataframe
vif = pd.DataFrame()
vif["VIF Factor"] = [VIF(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns

print(vif)
