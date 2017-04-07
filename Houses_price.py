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
    a = Dataframe.isnull().sum()[Dataframe.isnull().sum()>0]
    if len(a) == 0:
        return 'Null'
    else:
        return a
        
def barh_val(Data):
    '''
    horizontal bar graph with values at the end
    '''
    
    a= check_null(Data)
    max_width = int((np.logical_not(Data.isnull()).sum()+ Data.isnull().sum()).max()) 
    #is there any easier way to get count
    
    ax = a.plot.barh(color = '#FFA700', edgecolor = '#0F0A00', xlim = (0,max_width))
    ax.grid(False)
    ax.set_title('Barplot of columns with null values: number of null values')
    for p in ax.patches:
        width = p.get_width()
        percent = width/max_width*100
        ax.text(width+ 3 , p.get_y() + p.get_height()/2,  '{0}'.format(int(width)), color='#0F0A00',
                fontweight = 'bold', fontsize = 20, \
                verticalalignment ='center')
        ax.text(0 , p.get_y() + p.get_height()/2,  '{0}%'.format(int(percent)), color='#000000',
                fontweight = 'bold', fontsize = 9, \
                verticalalignment ='center')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 1.15, box.height])
barh_val(Data_num)
barh_val(Data_catg)
#%%
#Removing columns with too many null values (50% values)
rem = check_null(Data_catg)[check_null(Data_catg)/1460>.5].index
Data_catg_significant = Data_catg.drop(rem, axis = 1)
##convert_to_catg(col = 'LandContour')
#%%
#Creating Dummy Variables
def dummy_var(train_data, exception):
    '''
    Creating Dummy Variables
    '''
    train_data_1 = pd.DataFrame()
    for col in list(set(train_data.columns) - set(exception)):
        n = train_data[col].max()
        for i in range(n):
            col_name = col + '_' + str(i)
            train_data_1[col_name] = train_data[col].apply(lambda x: 1 if x == i else 0)
    return train_data_1
#%%
def multi_del(list1, texts):
    '''
    delete texts(list) from list1
    '''
    for i in texts:
        list1 = [ s.replace(i, '') for s in list1]
    return list1
#%%
# copied function: https://www.kaggle.com/neviadomski/house-prices-advanced-regression-techniques/how-to-get-to-top-25-with-simple-model-sklearn
# Prints R2 and RMSE scores
def get_score(prediction, lables):    
    print('R2: {}'.format(r2_score(prediction, lables)))
    print('RMSE: {}'.format(np.sqrt(mean_squared_error(prediction, lables))))

# Shows scores for train and validation sets    
def train_test(estimator, x_trn, x_tst, y_trn, y_tst):
    prediction_train = estimator.predict(x_trn)
    # Printing estimator
    print(estimator)
    # Printing train scores
    get_score(prediction_train, y_trn)
    prediction_test = estimator.predict(x_tst)
    # Printing test scores
    print("Test")
    get_score(prediction_test, y_tst)