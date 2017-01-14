# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 17:09:09 2017

@author: Nemish
"""
#House prices sample
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#%%
House_price_train = pd.read_csv('train.csv')
House_price_test = pd.read_csv('test.csv')

def convert_to_string(df, columns=None):
    '''
    converts a column to string
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
    c = HoP[col]
    d = HoP[['SalePrice']]
    plt.plot(c, d) # forgot how to plot in python
sales_vs_col(col = ['LandContour'])
#%%
convert_to_string(House_price_train, 'Id')
convert_to_string(House_price_test, 'Id')
summary = House_price_train.describe()