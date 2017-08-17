# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 16:46:41 2017

@author: Syuan-Ming
"""
#get_ipython().magic('matplotlib inline')
import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import (GridSearchCV, RandomizedSearchCV)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier # Used for imputing rare / missing values

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression # only model used for final submission

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from operator import itemgetter

os.chdir('E:\\Google Drive\\Python\\DS challenges\\Spanish_AB_test')        


#%% read train and test data
def read_table():
    print("Read user_table.csv...")
    df_user = pd.read_csv("./user_table.csv",
                       dtype={'country': np.str,                              
                              'age': np.int32,
                              'user_id': np.int32})
    print("Read test_table.csv...")
    df_test = pd.read_csv("./test_table.csv",
                       dtype={'country': np.str,                              
                              'conversion': np.int8,
                              'test': np.int8,
                              'user_id': np.int32},
                              parse_dates=['date'],
                              infer_datetime_format=True)
    
    return df_user,df_test

def process_table(df_user, df_test):
    print("Process tables...")
    df = pd.merge(df_test, df_user, how='left', on='user_id', left_index=True)
#    df.fillna(999, inplace=True)
    return df

def ttest(df): #perform two-tailed t-test 
    coversion_control = df[df['test']==0 & (df['country']!='Spain')]['conversion']
    coversion_test = df[df['test']==1 & (df['country']!='Spain')]['conversion']
    t,p = ttest_ind(coversion_control, coversion_test,equal_var=False)
    print('t-test p-value: %4.3e' % p)    
    return p

def ttest_single_country(df, country): #perform two-tailed t-test 
    coversion_control = df[df['test']==0 & (df['country']==country)]['conversion']
    coversion_test = df[df['test']==1 & (df['country']==country)]['conversion']
    t,p = ttest_ind(coversion_control, coversion_test,equal_var=False)
    print('country %s t-test p-value: %4.3e' % (country, p))          
    return p
#%% main script 
df_user,df_test = read_table()
df = process_table(df_user, df_test)
df_user.describe(include='all') #summerize the table            
df_test.describe(include='all') #summerize the table
df.describe(include='all') #summerize the table
# calculate average conversion rate for each country in control
df_country_convert = df[['country', 'conversion']][df['test']==0].groupby(['country']).mean().add_prefix('mean_').reset_index().sort_values('mean_conversion', ascending=False)            
ttest_result = ttest(df)
#%%
plt.figure(1, figsize=(8,6))
ax = sns.pointplot(x="date", y='conversion', data=df, hue='test',
                     ci=95, linewidth=1, errwidth=1, capsize=0.15, errcolor =[0,0,0])
plt.xticks(rotation=45)
#%%
plt.figure(2, figsize=(8,6))
ax = sns.countplot(x="country", hue='test',data=df, linewidth=1)
plt.xticks(rotation=45)
#%%
plt.figure(3, figsize=(8,6))
ax = sns.barplot(x="country", y='conversion', data=df, hue='test',
                     ci=95, linewidth=1, errwidth=1, capsize=0.15, errcolor =[0,0,0])
plt.xticks(rotation=45)
#%% 
country_list = df['country'].unique().tolist()
for country in country_list:
    ttest_single_country(df, country)
