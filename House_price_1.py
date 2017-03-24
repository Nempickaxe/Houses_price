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

plt.figure()
sns.boxplot(x = 'MSSubClass', y = 'SalePrice', data = House_price_train)
plt.figure()
sns.boxplot(x = 'Neighborhood', y = 'SalePrice', data = House_price_train)

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
House_cat = House_price_train.select_dtypes(exclude=numerics)
House_num = House_price_train.select_dtypes(include=numerics)

for i in House_cat.columns:
    plt.figure()
    sns.boxplot(x = i, y = 'SalePrice', data = House_price_train)
#%%
categ_1 = ['Alley', 'Neighborhood', 'Condition1', 'Condition2', 'HouseStyle', 'ExterQual', 'BsmtQual',\
            'BsmtCond', 'CentralAir', 'KitchenQual', 'GarageType', 'PavedDrive', 'SaleType']
categ_2 = ['MSSubClass', 'BldgType', 'Foundation', 'BsmtFinType1', 'Electrical', 'FireplaceQu', 'GarageFinish', 'PoolQC', \
            'SaleCondition']

for i in categ_1:
    plt.figure()
    sns.boxplot(x = i, y = 'SalePrice', data = House_price_train)

for i in categ_2:
    plt.figure()
    sns.boxplot(x = i, y = 'SalePrice', data = House_price_train)

#%% Reduce number of classes
for i in ['Condition1', 'Condition2']:
    plt.figure()
    sns.boxplot(x = i, y = 'SalePrice', data = House_price_train)
a = House_price_train[['Condition1', 'SalePrice']].groupby(['Condition1']).median().sort('SalePrice')

House_price_train["mod_Condition1"] = House_price_train["Condition1"].replace('Artery', 0)\
.replace(['Feedr', 'RRAe'], 1)\
.replace(['Norm', 'RRAn'], 2)\
.replace(['RRNe', 'PosN'], 3)\
.replace(['PosA', 'RRNn'], 4)

House_price_train["mod_Condition2"] = House_price_train["Condition2"].replace('Artery', 0)\
.replace(['Feedr', 'RRAe'], 1)\
.replace(['Norm', 'RRAn'], 2)\
.replace(['RRNe', 'PosN'], 3)\
.replace(['PosA', 'RRNn'], 4)

train_data['summ_Condition'] = (House_price_train["mod_Condition1"] + House_price_train["mod_Condition2"])

sns.boxplot(x = 'summ_Condition', y = 'SalePrice', data = House_price_train)
#%%
sns.boxplot(x = 'Alley', y = 'SalePrice', data = House_price_train)
plt.figure()
#%%
sns.boxplot(x = 'Neighborhood', y = 'SalePrice', data = House_price_train)
a = House_price_train[['Neighborhood', 'SalePrice']].groupby(['Neighborhood']).median().sort('SalePrice')

train_data["mod_Neighborhood"] = House_price_train["Neighborhood"].replace('MeadowV', 0)\
.replace(['IDOTRR', 'BrDale'], 1)\
.replace(['OldTown', 'Edwards', 'BrkSide'], 2)\
.replace(['Sawyer', 'Blueste', 'SWISU', 'NAmes', 'NPkVill', 'Mitchel'], 3)\
.replace(['SawyerW', 'Gilbert', 'NWAmes'], 4)\
.replace(['Blmngtn', 'CollgCr', 'ClearCr', 'Crawfor'], 5)\
.replace(['Veenker', 'Somerst', 'Timber'], 6)\
.replace(['StoneBr'], 7)\
.replace(['NoRidge'], 8)\
.replace(['NridgHt'], 9)

sns.boxplot(x = 'mod_Neighborhood', y = 'SalePrice', data = House_price_train, order = map(str, range(10)))
#%%
sns.boxplot(x = 'HouseStyle', y = 'SalePrice', data = House_price_train)
a = House_price_train[['HouseStyle', 'SalePrice']].groupby(['HouseStyle']).median().sort('SalePrice')

train_data["mod_HouseStyle"] = House_price_train["HouseStyle"].replace('1.5Unf', 0)\
.replace(['1.5Fin', 'SFoyer', '2.5Unf'], 1)\
.replace(['1Story', 'SLvl'], 2)\
.replace(['2Story', '2.5Fin'], 3)

sns.boxplot(x = 'mod_HouseStyle', y = 'SalePrice', data = House_price_train)
#%%
sns.boxplot(x = 'BsmtQual', y = 'SalePrice', data = House_price_train)
a = House_price_train[['BsmtQual', 'SalePrice']].groupby(['BsmtQual']).median().sort('SalePrice')

train_data["mod_BsmtQual"] = House_price_train["BsmtQual"].replace(np.nan, 0)\
.replace(['Fa', 'TA'], 1)\
.replace('Gd', 2)\
.replace('Ex', 3)

sns.boxplot(x = 'mod_BsmtQual', y = 'SalePrice', data = House_price_train) #nan's in original data means no basement
#%%
sns.boxplot(x = 'BsmtCond', y = 'SalePrice', data = House_price_train)
a = House_price_train[['BsmtCond', 'SalePrice']].groupby(['BsmtCond']).median().sort('SalePrice')

train_data["mod_BsmtCond"] = House_price_train["BsmtCond"].replace(np.nan, 0)\
.replace(['Po'], 1)\
.replace('Fa', 2)\
.replace(['TA', 'Gd'], 3)

sns.boxplot(x = 'mod_BsmtCond', y = 'SalePrice', data = House_price_train) # 0 means no basement
#%%
sns.boxplot(x = 'CentralAir', y = 'SalePrice', data = House_price_train)
#%%
sns.boxplot(x = 'KitchenQual', y = 'SalePrice', data = House_price_train)
#%%
sns.boxplot(x = 'GarageType', y = 'SalePrice', data = House_price_train)

i = 'GarageType'
a = House_price_train[[i, 'SalePrice']].groupby([i]).median().sort('SalePrice')

train_data["mod_GarageType"] = House_price_train["GarageType"].replace(np.nan, 0)\
.replace(['CarPort'], 1)\
.replace('Detchd', 2)\
.replace(['Basment', '2Types'], 3)\
.replace(['Attchd'], 3)\
.replace(['BuiltIn'], 4)

sns.boxplot(x = 'mod_GarageType', y = 'SalePrice', data = House_price_train)
#%%
sns.boxplot(x = 'PavedDrive', y = 'SalePrice', data = House_price_train)
#%%
sns.boxplot(x = 'SaleType', y = 'SalePrice', data = House_price_train)

i = 'SaleType'
a = House_price_train[[i, 'SalePrice']].groupby([i]).median().sort('SalePrice')

train_data["mod_SaleType"] = House_price_train["SaleType"].replace(['Oth', 'ConLI'] , 0)\
.replace(['COD', 'ConLD', 'ConLw'], 1)\
.replace('WD', 2)\
.replace(['CWD'], 3)\
.replace(['New'], 3)\
.replace(['Con'], 4)

sns.boxplot(x = 'mod_SaleType', y = 'SalePrice', data = House_price_train)
#%%
sns.boxplot(x = 'BldgType', y = 'SalePrice', data = House_price_train)

i = 'BldgType'
a = House_price_train[[i, 'SalePrice']].groupby([i]).median().sort('SalePrice')

train_data["mod_BldgType"] = House_price_train["BldgType"].replace(['2fmCon'] , 0)\
.replace(['Duplex', 'Twnhs'], 1)\
.replace('1Fam', 2)\
.replace(['TwnhsE'], 3)

sns.boxplot(x = 'mod_BldgType', y = 'SalePrice', data = House_price_train)
#%%
sns.boxplot(x = 'Foundation', y = 'SalePrice', data = House_price_train)

train_data["mod_Foundation"] = House_price_train["Foundation"].replace(["BrkTil", "CBlock", "Slab", "Stone", "Wood"] , 0)\
.replace(['PConc'], 1)

sns.boxplot(x = 'mod_Foundation', y = 'SalePrice', data = House_price_train)
#%%
sns.boxplot(x = 'BsmtFinType1', y = 'SalePrice', data = House_price_train)

i = 'BsmtFinType1'
a = House_price_train[[i, 'SalePrice']].groupby([i]).median().sort('SalePrice')

House_price_train["mod_BsmtFinType1"] = House_price_train["BsmtFinType1"].replace(np.nan, 0)\
.replace(['LwQ'] , 1)\
.replace(['BLQ'], 2)\
.replace('Rec', 3)\
.replace(['ALQ'], 4)\
.replace(['Unf'], 5)\
.replace(['GLQ'], 6)

House_price_train["mod_BsmtFinType2"] = House_price_train["BsmtFinType2"].replace(np.nan, 0)\
.replace(['LwQ'] , 1)\
.replace(['BLQ'], 2)\
.replace('Rec', 3)\
.replace(['ALQ'], 4)\
.replace(['Unf'], 5)\
.replace(['GLQ'], 6)

train_data["summ_BsmtFinType"] = (House_price_train["mod_BsmtFinType1"] +\
 House_price_train["mod_BsmtFinType2"]).astype(int)
 
train_data["summ_BsmtFinType"] = train_data["summ_BsmtFinType"].replace(["1", "2", "3", "4", "5", "6", "7", "8", "9"] , 1)\
.replace(['0'], 0)\
.replace(['10'], 2)\
.replace(['11'], 3)
 
sns.boxplot(x = 'summ_BsmtFinType', y = 'SalePrice', data = House_price_train)
#%%
sns.boxplot(x = 'Electrical', y = 'SalePrice', data = House_price_train)

i = 'Electrical'
a = House_price_train[[i, 'SalePrice']].groupby([i]).median().sort('SalePrice')

train_data["summ_Electrical"] = House_price_train["Electrical"].replace(["Mix"] , 0)\
.replace(['FuseP'], 1)\
.replace(['FuseF', "FuseA"], 2)\
.replace(['SBrkr'], 3)\

sns.boxplot(x = 'summ_Electrical', y = 'SalePrice', data = House_price_train)
#%%
sns.boxplot(x = 'FireplaceQu', y = 'SalePrice', data = House_price_train)

train_data["mod_FireplaceQu"] = House_price_train["FireplaceQu"].fillna("None")\
.map({"Ex": 3, "Gd": 3, "TA": 2, "Fa": 1, "Po": 1, "None": 0})

sns.boxplot(x = 'mod_FireplaceQu', y = 'SalePrice', data = House_price_train)
#%%
sns.boxplot(x = 'GarageFinish', y = 'SalePrice', data = House_price_train)
#%%
sns.boxplot(x = 'PoolQC', y = 'SalePrice', data = House_price_train)

train_data["mod_PoolQC"] = House_price_train["PoolQC"].fillna("None")\
.map({"Ex": 2, "Gd": 1, "TA": 1, "Fa": 1, "Po": 1, "None": 0})

sns.boxplot(x = 'mod_PoolQC', y = 'SalePrice', data = House_price_train)
#%%
sns.boxplot(x = 'SaleCondition', y = 'SalePrice', data = House_price_train)

train_data["mod_SaleCondition"] = House_price_train["SaleCondition"].fillna("None")\
.map({"Partial": 2, "Abnorml": 1, "Family": 1, "Alloca": 1, "Normal": 1, "AdjLand": 0, "None": 0})

sns.boxplot(x = 'mod_SaleCondition', y = 'SalePrice', data = House_price_train)

#%%
sns.boxplot(x = 'Alley', y = 'SalePrice', data = House_price_train)

train_data["mod_Alley"] = House_price_train["Alley"].fillna("None")\
.map({"Pave": 2, "Grvl": 1, "None":0})

sns.boxplot(x = 'mod_Alley', y = 'SalePrice', data = train_data)
#%%
sns.boxplot(x = 'ExterQual', y = 'SalePrice', data = House_price_train)

train_data["mod_ExterQual"] = House_price_train["ExterQual"].fillna("None")\
.map({"Ex": 3, "Gd": 2, "TA": 1, "Fa": 0, "Po": 0})

sns.boxplot(x = 'mod_ExterQual', y = 'SalePrice', data = train_data)
#%%
sns.boxplot(x = 'CentralAir', y = 'SalePrice', data = House_price_train)

train_data["mod_CentralAir"] = House_price_train["CentralAir"].fillna("None")\
.map({"Y": 1, "N": 0})

sns.boxplot(x = 'mod_CentralAir', y = 'SalePrice', data = train_data)
#%%
sns.boxplot(x = 'KitchenQual', y = 'SalePrice', data = House_price_train)

train_data["mod_KitchenQual"] = House_price_train["KitchenQual"].fillna("None")\
.map({"Ex": 3, "Gd": 2, "TA": 1, "Fa": 0, "Po": 0})

sns.boxplot(x = 'mod_KitchenQual', y = 'SalePrice', data = train_data)
#%%
sns.boxplot(x = 'PavedDrive', y = 'SalePrice', data = House_price_train)

train_data["mod_PavedDrive"] = House_price_train["PavedDrive"].fillna("None")\
.map({"Y": 2, "P": 1, "N": 0})

sns.boxplot(x = 'mod_PavedDrive', y = 'SalePrice', data = train_data)
#%%
sns.boxplot(x = 'GarageFinish', y = 'SalePrice', data = House_price_train)

train_data["mod_GarageFinish"] = House_price_train["GarageFinish"].fillna("None")\
.map({"Fin": 3, "RFn":2, "Unf": 1, "None": 0})

sns.boxplot(x = 'mod_GarageFinish', y = 'SalePrice', data = train_data)
#%%
sns.boxplot(x = 'OverallQual', y = 'SalePrice', data = House_price_train)

train_data["mod_OverallQual"] = (House_price_train["OverallQual"]-1)
#%%
def check_null(Dataframe):
    '''
    Returns columns with number of nulls
    '''
    return Dataframe.isnull().sum()[Dataframe.isnull().sum()>0]
    
corrmat = train_data.corr().round(2)

sns.heatmap(corrmat, annot=True)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
#%%
col = [w.replace('mod_', '').replace('summ_', '') for w in train_data.columns]
    
a = [x for x in House_cat.columns if x not in col]

for i in a:
      if i !=  'SalePrice': 
        plt.figure()
        sns.boxplot(x = i, y = 'SalePrice', data = House_price_train)
#%%     
b = ['MSZoning', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'RoofMatl', 'ExterCond',  'LotConfig',
     'BsmtExposure', 'HeatingQC']
     
sns.boxplot(x = 'MSZoning', y = 'SalePrice', data = House_price_train)
           
train_data["mod_MSZoning"] = House_price_train["MSZoning"].fillna("None")\
.map({"FV":3, "RL": 2, "RH":1, "RM": 1, "C (all)": 0})

sns.boxplot(x = 'mod_MSZoning', y = 'SalePrice', data = train_data)
#%%
i = 'Exterior1st'
a = House_price_train[[i, 'SalePrice']].groupby([i]).median().sort('SalePrice')

House_price_train["mod_Exterior1st"] = House_price_train["Exterior1st"].fillna("None")\
.map({"ImStucc":7, "Stone":7, "CemntBd":6, "VinylSd":5, "Plywood":4,\
    "BrkFace":4, "HdBoard":3, "Stucco":3, "MetalSd":2, "Wd Sdng":2,\
    "WdShing":2, "AsbShng": 1, "CBlock":1, "AsphShn": 1, "BrkComm": 0})

House_price_train["mod_Exterior2nd"] = House_price_train["Exterior2nd"].fillna("None")\
.map({"ImStucc":7, "Stone":7, "CemntBd":6, "VinylSd":5, "Plywood":4,\
    "BrkFace":4, "HdBoard":3, "Stucco":3, "MetalSd":2, "Wd Sdng":2,\
    "WdShing":2, "AsbShng": 1, "CBlock":1, "AsphShn": 1, "BrkComm": 0})

#train_data['summ_Exterior'] = House_price_train["mod_Exterior1st"] + House_price_train["mod_Exterior2nd"]

sns.boxplot(x = 'summ_Exterior', y = 'SalePrice', data = train_data)
#%%
sns.boxplot(x = 'MasVnrType', y = 'SalePrice', data = House_price_train)

train_data["mod_MasVnrType"] = House_price_train["MasVnrType"].fillna("None")\
.map({"Stone": 2, "BrkFace":1, "BrkCmn": 0, "None": 0})

sns.boxplot(x = 'mod_MasVnrType', y = 'SalePrice', data = train_data)
#%%
sns.boxplot(x = 'RoofMatl', y = 'SalePrice', data = House_price_train)

train_data["mod_RoofMatl"] = House_price_train["RoofMatl"].fillna("None")\
.map({"WdShngl": 1,"CompShg":0, "Metal":0, "WdShake":0, "Membran":0,\
      "Tar&Grv":0, "Roll":0, "ClyTile":0})
      
sns.boxplot(x = 'mod_RoofMatl', y = 'SalePrice', data = train_data)
#%%
sns.boxplot(x = 'ExterCond', y = 'SalePrice', data = House_price_train)

train_data["mod_ExterCond"] = House_price_train["ExterCond"].fillna("None")\
.map({"Ex":1, "Gd":1, "TA":1, "Fa":0,\
      "Po":0})
      
sns.boxplot(x = 'mod_ExterCond', y = 'SalePrice', data = train_data)
#%%
sns.boxplot(x = 'BsmtExposure', y = 'SalePrice', data = House_price_train)

train_data["mod_BsmtExposure"] = House_price_train["BsmtExposure"].fillna("None")\
.map({"Av":2, "Gd":2, "Mn":2, "No":1,\
      "None":0})

sns.boxplot(x = 'mod_BsmtExposure', y = 'SalePrice', data = train_data)
#%%
sns.boxplot(x = 'HeatingQC', y = 'SalePrice', data = House_price_train)

train_data["mod_HeatingQC"] = House_price_train["HeatingQC"].fillna("None")\
.map({"Ex":1, "Gd":0, "TA":0, "Fa":0,\
      "Po":0})

sns.boxplot(x = 'mod_HeatingQC', y = 'SalePrice', data = train_data)
#%%
sns.boxplot(x = 'mod_BsmtExposure', y = 'SalePrice', data = train_data)
