# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 22:40:40 2017

@author: Nemish
"""

#House prices sample
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
%matplotlib qt
#%%
House_price_train = pd.read_csv('train.csv')
del House_price_train['Id']
#%%
sns.distplot(House_price_train['SalePrice'])
House_price_train.plot.scatter( x = 'LotArea', y = 'SalePrice')

sns.boxplot(x = 'Neighborhood', y = 'SalePrice', data = House_price_train)