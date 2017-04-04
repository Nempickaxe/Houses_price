# -*- coding: utf-8 -*-
"""
Created on Tue Apr 04 21:24:27 2017

@author: Nemish
"""

#House prices sample
import pandas as pd
import numpy as np

#%%
iitial = 0
if(iitial == 0):
    House_price_train = pd.read_csv('train.csv')
    House_price_train.index = House_price_train['Id']
    #no_use
    del House_price_train['Id']
    #Multicollinearity_cat
    del House_price_train['ExterQual']
    del House_price_train['GarageType']
    del House_price_train['KitchenQual']
    #Multicollinearity_num
    del House_price_train['GarageArea']
    del House_price_train['TotalBsmtSF']
    del House_price_train['TotRmsAbvGrd']
    del House_price_train['GarageYrBlt']
#    #No information gain
#    del House_price_train['YrSold']
#    del House_price_train['MoSold']
#    del House_price_train['MiscVal']
#    del House_price_train['PoolArea']
#    del House_price_train['WoodDeckSF']
#    del House_price_train['KitchenAbvGr']
    
    iitial+=1
