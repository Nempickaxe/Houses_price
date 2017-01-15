# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 17:09:09 2017

@author: Nemish
"""
#House prices sample
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
#%%
House_price_train = pd.read_csv('train.csv')
House_price_test = pd.read_csv('test.csv')
HoP = House_price_train[:] # external copy, just for reference
#%%
def convert_to_string(df, columns=None):
    '''
    Converts a column to string
    '''
    df[columns] =  map(lambda x: str(x), df[columns])

def no_null_objects(data, columns=None):
    """
    Returns rows with no NaNs
    """
    if columns is None:
        columns = data.columns
    return data[np.logical_not(np.any(data[columns].isnull().values, axis=1))]

def sales_vs_col(col , HoP = House_price_train):
    '''
    Plot column v/s SalePrice on y-axis
    '''
    sns.boxplot(x = col, y = 'SalePrice', data = HoP)
sales_vs_col(col = 'LandContour')
#%%
def convert_to_catg(col, HoP = House_price_train):
   '''
   Converts a column to category data type
   '''
   HoP[col] = HoP[col].astype('category')

#%%
convert_to_string(House_price_train, 'Id')
convert_to_string(House_price_test, 'Id')
summary = House_price_train.describe()

# categorical variables
cat_var = House_price_train.dtypes[House_price_train.dtypes == object].index.tolist()

#Data separated into categorical and numerical df
Data_catg = House_price_train[cat_var]
Data_num = House_price_train[list(set(HoP.columns)-set(cat_var))]

#%%
#Checking if null values present
def check_null(Dataframe):
    '''
    Returns columns with number of nulls
    '''
    return Dataframe.isnull().sum()[Dataframe.isnull().sum()>0]
def barh_val(a):
    '''
    horizontal bar graph with values at the end
    '''
    ax = a.plot.barh(color = '#FFA700', edgecolor = '#0F0A00') 
    for p in ax.patches:
        width = p.get_width()
        ax.text(width+ 3 , p.get_y() + p.get_height()/2,  '%d'%width, color='#0F0A00',
                fontweight = 'bold', fontsize = 20, \
                verticalalignment ='center')
barh_val(check_null(Data_num))
barh_val(check_null(Data_catg))
#%%
##convert_to_catg(col = 'LandContour')