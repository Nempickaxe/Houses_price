# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 14:50:17 2017

@author: nkanwar
"""
import pandas as pd
import numpy as np
House_price_train = pd.read_csv('train_data.csv')
House_price_test = pd.read_csv('test_data.csv')

House_price_train = House_price_train.rename(columns = {'Unnamed: 0':'Id'})
House_price_test = House_price_test.rename(columns = {'Unnamed: 0':'Id'})
#%%
from sklearn.ensemble import GradientBoostingRegressor  #GBM algorithm
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

train_x = House_price_train.iloc[:, :-1].as_matrix()
train_y = House_price_train.iloc[:, -1].as_matrix()
test_x = House_price_test.as_matrix()

# The error metric: RMSE on the log of the sale prices.
from sklearn.metrics import mean_squared_error
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
#%%
gbm0 = GradientBoostingRegressor(random_state=10)
gbm0.fit(train_x, train_y)

print 'R2:', gbm0.score(train_x, train_y), 'rmse:', rmse(np.log1p(train_y), np.log1p(gbm0.predict(train_x)))
#%%
'''
https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
min_sample_split(~0.5-1% of total values): 8
max_depth = 7 (5-8)
subsample = 0.8
learning_rate = 0.1
max_features = 'sqrt'
loss = going with 'huber'
min_samples_leaf: 50 Defines the minimum samples (or observations) required in a terminal node or leaf.
'''
param_test1 = {'n_estimators':range(20,200,100)}
gsearch1 = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, max_depth=7, max_features='sqrt',
                                               min_samples_leaf=15, min_samples_split=8, loss='huber', subsample = 0.8, random_state = 10),
                                                               param_grid = param_test1,
                                                               scoring= None ,
                                                               n_jobs=4,iid=False, cv=5)
gsearch1.fit(train_x, train_y)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_