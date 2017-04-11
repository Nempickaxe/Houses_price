# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 14:02:49 2017

@author: nkanwar
"""
#House prices sample
import pandas as pd
import numpy as np
#%%
House_price_train = pd.read_csv('train.csv')
House_price_test = pd.read_csv('test.csv')

House_price_train.index = House_price_train.Id
House_price_test.index = House_price_test.Id
#%%
#Outlier removal
House_price_train.drop([1299, 935, 314, 336, 250, 707, 692, 1183, 524, 497], inplace = True)
#%%
lot_frontage_by_neighborhood = House_price_train["LotFrontage"].groupby(House_price_train["Neighborhood"])
#%%
def munge(df):
    all_df = pd.DataFrame(index = df.index)
    
    all_df["LotFrontage"] = df["LotFrontage"]   
    for key, group in lot_frontage_by_neighborhood:
        idx = (df["Neighborhood"] == key) & (df["LotFrontage"].isnull())
        all_df.loc[idx, "LotFrontage"] = group.median()
    
    #BsmtFinSF        
    all_df["BsmtFinSF1"] = df["BsmtFinSF1"]
    all_df["BsmtFinSF1"].fillna(0, inplace=True)
    all_df["BsmtFinSF2"] = df["BsmtFinSF2"]
    all_df["BsmtFinSF2"].fillna(0, inplace=True)
    all_df["BsmtUnfSF"] = df["BsmtUnfSF"]
    all_df["BsmtUnfSF"].fillna(0, inplace=True)
    all_df['summ_BsmtSF'] = all_df['BsmtUnfSF'] + all_df['BsmtFinSF2'] + all_df['BsmtFinSF1']
    all_df['summ_livBsSF'] = df['GrLivArea'] + all_df['summ_BsmtSF']
    all_df.drop(['BsmtUnfSF', 'BsmtFinSF2', 'BsmtFinSF1', 'summ_BsmtSF'], axis=1, inplace=True)

    #Bathrooms summation
    all_df["BsmtHalfBath"] = df["BsmtHalfBath"]
    all_df["BsmtHalfBath"].fillna(0, inplace=True)
    all_df["BsmtFullBath"] = df["BsmtFullBath"]
    all_df["BsmtFullBath"].fillna(0, inplace=True)
    all_df["FullBath"] = df["FullBath"]
    all_df["FullBath"].fillna(0, inplace=True)
    all_df["HalfBath"] = df["HalfBath"]
    all_df["HalfBath"].fillna(0, inplace=True)
    all_df['summ_Bathrooms'] = all_df['BsmtHalfBath'] + all_df['BsmtFullBath']+\
                                all_df['FullBath'] + all_df['HalfBath']
    all_df.drop(['BsmtHalfBath', 'BsmtFullBath', 'FullBath', 'HalfBath'], axis=1, inplace=True)                            
    
    all_df["LotArea"] = df["LotArea"]

    all_df["MasVnrArea"] = df["MasVnrArea"]
    all_df["MasVnrArea"].fillna(0, inplace=True)    
    
    all_df["BedroomAbvGr"] = df["BedroomAbvGr"] 
    
    all_df["GarageCars"] = df["GarageCars"]
    all_df["GarageCars"].fillna(0, inplace=True)
    
    
    #################################################################################
    #Categorical variables
    #Conditions
    all_df["mod_Condition1"] = df["Condition1"].replace('Artery', 0)\
    .replace(['Feedr', 'RRAe'], 1)\
    .replace(['Norm', 'RRAn'], 2)\
    .replace(['RRNe', 'PosN'], 3)\
    .replace(['PosA', 'RRNn'], 4)
    
    all_df["mod_Condition2"] = df["Condition2"].replace('Artery', 0)\
    .replace(['Feedr', 'RRAe'], 1)\
    .replace(['Norm', 'RRAn'], 2)\
    .replace(['RRNe', 'PosN'], 3)\
    .replace(['PosA', 'RRNn'], 4)
    
    all_df['summ_Condition'] = (all_df["mod_Condition1"] + all_df["mod_Condition2"])
    all_df['summ_Condition'] = all_df['summ_Condition'].replace([0,1], 0)\
    .replace([2,3], 1)\
    .replace([4,5], 2)\
    .replace([5,6,7,8,9], 3)
    all_df.drop(['mod_Condition1', 'mod_Condition2'], axis=1, inplace=True)                            

    #Neighbourhood
    all_df["mod_Neighborhood"] = df["Neighborhood"].replace('MeadowV', 0)\
    .replace(['IDOTRR', 'BrDale'], 1)\
    .replace(['OldTown', 'Edwards', 'BrkSide'], 2)\
    .replace(['Sawyer', 'Blueste', 'SWISU', 'NAmes', 'NPkVill', 'Mitchel'], 3)\
    .replace(['SawyerW', 'Gilbert', 'NWAmes'], 4)\
    .replace(['Blmngtn', 'CollgCr', 'ClearCr', 'Crawfor'], 5)\
    .replace(['Veenker', 'Somerst', 'Timber'], 6)\
    .replace(['StoneBr'], 7)\
    .replace(['NoRidge'], 8)\
    .replace(['NridgHt'], 9)
    
    #House Style
    all_df["mod_HouseStyle"] = df["HouseStyle"].replace('1.5Unf', 0)\
    .replace(['1.5Fin', 'SFoyer', '2.5Unf'], 1)\
    .replace(['1Story', 'SLvl'], 2)\
    .replace(['2Story', '2.5Fin'], 3)
    
    #Bsmt Quality and Condition
    all_df["mod_BsmtQual"] = df["BsmtQual"].replace(np.nan, 0)\
    .replace(['Fa', 'TA'], 1)\
    .replace('Gd', 2)\
    .replace('Ex', 3)
    
    all_df["mod_BsmtCond"] = df["BsmtCond"].replace(np.nan, 0)\
    .replace(['Po'], 1)\
    .replace('Fa', 2)\
    .replace(['TA', 'Gd'], 3)
    
    #Saletype
    all_df["mod_SaleType"] = df["SaleType"].fillna("None")\
    .map({'Con':4, 'New':3, 'CWD':3, 'WD':2, 'COD':1, 'ConLD':1, 'ConLw':1, 'Oth':0, 'ConLI':0, 'None':0})
    
    #Building Type
    all_df["mod_BldgType"] = df["BldgType"].replace(['2fmCon'] , 0)\
    .replace(['Duplex', 'Twnhs'], 1)\
    .replace('1Fam', 2)\
    .replace(['TwnhsE'], 3)
    
    #Foundation
    all_df["mod_Foundation"] = df["Foundation"].replace(["BrkTil", "CBlock", "Slab", "Stone", "Wood"] , 0)\
    .replace(['PConc'], 1)
    
    #Basement Finish type
    all_df["mod_BsmtFinType1"] = df["BsmtFinType1"].replace(np.nan, 0)\
    .replace(['LwQ'] , 1)\
    .replace(['BLQ'], 2)\
    .replace('Rec', 3)\
    .replace(['ALQ'], 4)\
    .replace(['Unf'], 5)\
    .replace(['GLQ'], 6)
    all_df["mod_BsmtFinType2"] = df["BsmtFinType2"].replace(np.nan, 0)\
    .replace(['LwQ'] , 1)\
    .replace(['BLQ'], 2)\
    .replace('Rec', 3)\
    .replace(['ALQ'], 4)\
    .replace(['Unf'], 5)\
    .replace(['GLQ'], 6)
    all_df["summ_BsmtFinType"] = (all_df["mod_BsmtFinType1"] + all_df["mod_BsmtFinType2"]).astype(int)
    all_df.drop(['mod_BsmtFinType1', 'mod_BsmtFinType2'], axis=1, inplace=True)     
    all_df["summ_BsmtFinType"] = all_df["summ_BsmtFinType"].replace(["1", "2", "3", "4", "5", "6", "7", "8", "9"] , 1)\
    .replace(['0'], 0)\
    .replace(['10'], 2)\
    .replace(['11'], 3)
    
    all_df["mod_Electrical"] = df["Electrical"].fillna("None")\
    .map({'SBrkr':3, 'FuseF': 2, 'FuseA': 2, 'FuseP': 1, 'Mix': 0, 'None': 0})
    
    all_df["mod_FireplaceQu"] = df["FireplaceQu"].fillna("None")\
    .map({"Ex": 3, "Gd": 3, "TA": 2, "Fa": 1, "Po": 1, "None": 0})
    
    all_df["mod_PoolQC"] = df["PoolQC"].fillna("None")\
    .map({"Ex": 2, "Gd": 1, "TA": 1, "Fa": 1, "Po": 1, "None": 0})
    
    all_df["mod_SaleCondition"] = df["SaleCondition"].fillna("None")\
    .map({"Partial": 2, "Abnorml": 1, "Family": 1, "Alloca": 1, "Normal": 1, "AdjLand": 0, "None": 0})
    
    all_df["mod_Alley"] = df["Alley"].fillna("None")\
    .map({"Pave": 2, "Grvl": 1, "None":0})
    
    #all_df["mod_ExterQual"] = Df["ExterQual"].fillna("None")\
    #.map({"Ex": 3, "Gd": 2, "TA": 1, "Fa": 0, "Po": 0})
    
    all_df["mod_CentralAir"] = df["CentralAir"].fillna("None")\
    .map({"Y": 1, "N": 0})
    
    #all_df["mod_KitchenQual"] = Df["KitchenQual"].fillna("None")\
    #.map({"Ex": 3, "Gd": 2, "TA": 1, "Fa": 0, "Po": 0})
    
    all_df["mod_PavedDrive"] = df["PavedDrive"].fillna("None")\
    .map({"Y":2, "P": 1, "N": 0})
    
    all_df["mod_GarageFinish"] = df["GarageFinish"].fillna("None")\
    .map({"Fin": 3, "RFn":2, "Unf": 1, "None": 0})
    
    all_df["mod_OverallQual"] = (df["OverallQual"])
    
    all_df["mod_MSZoning"] = df["MSZoning"].fillna("None")\
    .map({"FV":3, "RL": 2, "RH":1, "RM": 1, "C (all)": 0, "None":0})
    
    #all_df["mod_Exterior1st"] = Df["Exterior1st"].fillna("None")\
    #.map({"ImStucc":7, "Stone":7, "CemntBd":6, "VinylSd":5, "Plywood":4,\
    #    "BrkFace":4, "HdBoard":3, "Stucco":3, "MetalSd":2, "Wd Sdng":2,\
    #    "WdShing":2, "AsbShng": 1, "CBlock":1, "AsphShn": 1, "BrkComm": 0})
        
    all_df["mod_MasVnrType"] = df["MasVnrType"].fillna("None")\
    .map({"Stone": 2, "BrkFace":1, "BrkCmn": 0, "None": 0})
    
    all_df["mod_RoofMatl"] = df["RoofMatl"].fillna("None")\
    .map({"WdShngl": 1,"CompShg":0, "Metal":0, "WdShake":0, "Membran":0,\
          "Tar&Grv":0, "Roll":0, "ClyTile":0})
          
    all_df["mod_ExterCond"] = df["ExterCond"].fillna("None")\
    .map({"Ex":1, "Gd":1, "TA":1, "Fa":0,\
          "Po":0})
    
    all_df["mod_BsmtExposure"] = df["BsmtExposure"].fillna("None")\
    .map({"Av":2, "Gd":2, "Mn":2, "No":1,\
          "None":0})
    
    all_df["mod_HeatingQC"] = df["HeatingQC"].fillna("None")\
    .map({"Ex":1, "Gd":0, "TA":0, "Fa":0,\
          "Po":0})
    
    all_df["mod_MSSubClass"] = df["MSSubClass"].replace([20,  70,  50, 190, 150,  45,  90, 120,  85,  80, 160,  75,
           180,  40] , 1)\
    .replace([30], 0)\
    .replace([60], 2)
    
    all_df['mod_OverallCond'] = df['OverallCond'].apply(lambda x: 1 if x >= 5 else 0)
    
    all_df['mod_OpenPorchSF'] = df['OpenPorchSF'].apply(lambda x: 0.2 if x != 0 else 0)
    all_df['mod_EnclosedPorch'] = df['EnclosedPorch'].apply(lambda x: 0.3 if x != 0 else 0)
    all_df['mod_3SsnPorch'] = df['3SsnPorch'].apply(lambda x: 0.7 if x != 0 else 0)
    all_df['mod_ScreenPorch'] = df['ScreenPorch'].apply(lambda x: 0.11 if x != 0 else 0)
    all_df['summ_porch_cond'] = all_df['mod_OpenPorchSF'] + all_df['mod_EnclosedPorch'] + \
                                 all_df['mod_3SsnPorch'] + all_df['mod_ScreenPorch']
    all_df["mod_summ_porch_cond"] = all_df['summ_porch_cond'].apply(lambda x: 1 if x in (.2, .31) else 0)
    all_df.drop(['summ_porch_cond', 'mod_OpenPorchSF', 'mod_EnclosedPorch', 'mod_3SsnPorch', 'mod_ScreenPorch'], axis=1, inplace=True)

    all_df['mod_Fireplaces'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
    
    all_df['mod_LowQualFinSF'] = df['LowQualFinSF'].apply(lambda x: 1 if x > 0 else 0)

    ##
    def year_era(x):
        if x < 1950:
            return 0
        elif x < 2000:
            return 1
        else:
            return 2
    all_df['era_YearBuilt'] = df['YearBuilt'].apply(year_era)
    all_df['era_YearRemodAdd'] = df['YearRemodAdd'].apply(year_era)
    all_df['mod_remod'] = np.where(all_df['era_YearBuilt'] == all_df['era_YearRemodAdd'], 0, 1)
    all_df.drop(['era_YearRemodAdd'], axis=1, inplace=True)
    
    return all_df

munge_train = munge(House_price_train)
munge_test = munge(House_price_test)
#%%
# Transform the skewed numeric features by taking log(feature + 1).
# This will make the features more normal.
from scipy.stats import skew

numeric_features = ['LotFrontage', 'BedroomAbvGr', 'summ_Bathrooms', 'GarageCars',
                    'summ_livBsSF', 'LotArea', 'MasVnrArea']
skewed = munge_train[numeric_features].apply(lambda x: skew(x.dropna().astype(float)))
skewed = skewed[skewed > 0.75]
skewed = skewed.index

munge_train[skewed] = np.log1p(munge_train[skewed])
munge_test[skewed] = np.log1p(munge_test[skewed])
#%%
#scale the data.   
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(munge_train[numeric_features])

scaled = scaler.transform(munge_train[numeric_features])
for i, col in enumerate(numeric_features):
    munge_train[col] = scaled[:, i]

scaled = scaler.transform(munge_test[numeric_features])
for i, col in enumerate(numeric_features):
    munge_test[col] = scaled[:, i]
#%%
label_df = pd.DataFrame(index = House_price_train.index, columns=["SalePrice"])
label_df["SalePrice"] = np.log(House_price_train["SalePrice"])
#%%
#FillNa's with median
munge_train.fillna(munge_train.median(), inplace=True)
munge_test.fillna(munge_test.median(), inplace=True)
#Dummy Variable
# Convert categorical features using one-hot encoding.
cat_features = list(set(list(munge_test))-set(numeric_features))

def onehot(onehot_df, df, column_name, fill_na):
    onehot_df[column_name] = df[column_name]
    if fill_na is not None:
        onehot_df[column_name].fillna(fill_na, inplace=True)

    dummies = pd.get_dummies(onehot_df[column_name], prefix="_" + column_name, drop_first = False)
    
    # Dropping one of the columns actually made the results slightly worse.
    # if drop_name is not None:
    #     dummies.drop(["_" + column_name + "_" + drop_name], axis=1, inplace=True)

    onehot_df = onehot_df.join(dummies)
    onehot_df = onehot_df.drop([column_name], axis=1)
    return onehot_df
    
def munge_onehot(df):
    onehot_df = pd.DataFrame(index = df.index)
    
    onehot_df = onehot(onehot_df, df, 'mod_BsmtExposure', None)
    onehot_df = onehot(onehot_df, df, 'mod_ExterCond', None)
    onehot_df = onehot(onehot_df, df, 'mod_GarageFinish', None)
    onehot_df = onehot(onehot_df, df, 'mod_BsmtCond', None)
    onehot_df = onehot(onehot_df, df, 'mod_CentralAir', None)
    onehot_df = onehot(onehot_df, df, 'summ_BsmtFinType', None)
    onehot_df = onehot(onehot_df, df, 'mod_BldgType', None)
    onehot_df = onehot(onehot_df, df, 'mod_Neighborhood', None)
    onehot_df = onehot(onehot_df, df, 'mod_remod', None)
    onehot_df = onehot(onehot_df, df, 'mod_MSZoning', None)
    onehot_df = onehot(onehot_df, df, 'mod_FireplaceQu', None)
    onehot_df = onehot(onehot_df, df, 'mod_OverallCond', None)
    onehot_df = onehot(onehot_df, df, 'era_YearBuilt', None)
    onehot_df = onehot(onehot_df, df, 'mod_Foundation', None)
    onehot_df = onehot(onehot_df, df, 'mod_summ_porch_cond', None)
    onehot_df = onehot(onehot_df, df, 'mod_SaleCondition', None)
    onehot_df = onehot(onehot_df, df, 'mod_Alley', None)
    onehot_df = onehot(onehot_df, df, 'mod_LowQualFinSF', None)
    onehot_df = onehot(onehot_df, df, 'mod_SaleType', None)
    onehot_df = onehot(onehot_df, df, 'mod_HeatingQC', None)
    onehot_df = onehot(onehot_df, df, 'mod_BsmtQual', None)
    onehot_df = onehot(onehot_df, df, 'summ_Condition', None)
    onehot_df = onehot(onehot_df, df, 'mod_RoofMatl', None)
    onehot_df = onehot(onehot_df, df, 'mod_OverallQual', None)
    onehot_df = onehot(onehot_df, df, 'mod_Electrical', None)
    onehot_df = onehot(onehot_df, df, 'mod_MasVnrType', None)
    onehot_df = onehot(onehot_df, df, 'mod_MSSubClass', None)
    onehot_df = onehot(onehot_df, df, 'mod_PavedDrive', None)
    onehot_df = onehot(onehot_df, df, 'mod_Fireplaces', None)
    onehot_df = onehot(onehot_df, df, 'mod_HouseStyle', None)
    onehot_df = onehot(onehot_df, df, 'mod_PoolQC', None)
    
    return onehot_df
onehot_df = munge_onehot(munge_train)
train_df_munged = onehot_df.join(munge_train[numeric_features])

onehot_df = munge_onehot(munge_test)
test_df_munged = onehot_df.join(munge_test[numeric_features])

#need to remove '_mod_Electrical_0' from train data to prevent overfitting
train_df_munged.drop('_mod_Electrical_0', axis=1, inplace=True)
#%%
#Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor  #GBM algorithm
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

train_x = train_df_munged.as_matrix()
train_y = label_df.as_matrix()
test_x = test_df_munged.as_matrix()

# The error metric: RMSE on the log of the sale prices.
from sklearn.metrics import mean_squared_error
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

#%%
#Baseline Model
gbm0 = GradientBoostingRegressor(random_state=10)
gbm0.fit(train_x, train_y)

print 'R2:', gbm0.score(train_x, train_y), 'rmse:', rmse(train_y, gbm0.predict(train_x))
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
param_test1 = {'n_estimators':range(20,500,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, max_depth=7, max_features='sqrt',
                                               min_samples_leaf=15, min_samples_split=8, loss='huber', subsample = 0.8, random_state = 10),
                                                               param_grid = param_test1,
                                                               scoring= None ,
                                                               n_jobs=4,iid=False, cv=5)
gsearch1.fit(train_x, train_y)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
'''
selected n_estimator = 350
'''
#%%
param_test2 = {'max_depth':range(5,16,2), 'min_samples_split':range(5,16,2)}
gsearch2 = GridSearchCV(estimator = GradientBoostingRegressor(n_estimators = 350, learning_rate=0.1, max_features='sqrt',
                                               min_samples_leaf=15, loss='huber', subsample = 0.8, random_state = 10),
                                                               param_grid = param_test2,
                                                               scoring= None ,
                                                               n_jobs=4,iid=False, cv=5)
gsearch2.fit(train_x, train_y)
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_
'''
both max_depth and min_samples_split converged at 5 an extremities
'''
#%%
param_test3 = {'max_depth':range(2,6,1), 'min_samples_split':range(2,6,1)}
gsearch3 = GridSearchCV(estimator = GradientBoostingRegressor(n_estimators = 350, learning_rate=0.1, max_features='sqrt',
                                               min_samples_leaf=15, loss='huber', subsample = 0.8, random_state = 10),
                                                               param_grid = param_test3,
                                                               scoring= None ,
                                                               n_jobs=4,iid=False, cv=5)
gsearch3.fit(train_x, train_y)
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
'''
selected max_depth: 5, min_samples_split: 2
'''
#%%
param_test4 = {'min_samples_leaf':range(30,71,10)}
gsearch4 = GridSearchCV(estimator = GradientBoostingRegressor(n_estimators = 350, learning_rate=0.1, max_depth=5, max_features='sqrt',
                                               min_samples_split=2, loss='huber', subsample = 0.8, random_state = 10),
                                                               param_grid = param_test4,
                                                               scoring= None ,
                                                               n_jobs=4,iid=False, cv=5)
gsearch4.fit(train_x, train_y)
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_
'''
converged at extremity, min_samples_leaf: 30
'''
#%%
param_test5 = {'min_samples_leaf':range(10,31,10)}
gsearch5 = GridSearchCV(estimator = GradientBoostingRegressor(n_estimators = 350, learning_rate=0.1, max_depth=5, max_features='sqrt',
                                               min_samples_split=2, loss='huber', subsample = 0.8, random_state = 10),
                                                               param_grid = param_test5,
                                                               scoring= None ,
                                                               n_jobs=4,iid=False, cv=5)
gsearch5.fit(train_x, train_y)
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_
'''
selected min_samples_leaf: 20
'''
#%%
param_test6 = {'max_features':range(7,12,1)}
gsearch6 = GridSearchCV(estimator = GradientBoostingRegressor(n_estimators = 350, learning_rate=0.1, max_depth=5,
                                               min_samples_leaf= 20, min_samples_split=2, loss='huber', subsample = 0.8, random_state = 10),
                                                               param_grid = param_test6,
                                                               scoring= None ,
                                                               n_jobs=4,iid=False, cv=5)
gsearch6.fit(train_x, train_y)
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_
'''
extremity max_features: 9
'''
#%%
param_test7 = {'max_features':range(2,8,1)}
gsearch7 = GridSearchCV(estimator = GradientBoostingRegressor(n_estimators = 350, learning_rate=0.1, max_depth=5,
                                               min_samples_leaf= 20, min_samples_split=2, loss='huber', subsample = 0.8, random_state = 10),
                                                               param_grid = param_test7,
                                                               scoring= None ,
                                                               n_jobs=4,iid=False, cv=5)
gsearch7.fit(train_x, train_y)
gsearch7.grid_scores_, gsearch7.best_params_, gsearch7.best_score_
'''
selected max_features: 7
'''
#%%
param_test8 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
gsearch8 = GridSearchCV(estimator = GradientBoostingRegressor(n_estimators = 350, learning_rate=0.1, max_depth=5, max_features= 7,
                                               min_samples_leaf= 20, min_samples_split=2, loss='huber', random_state = 10),
                                                               param_grid = param_test8,
                                                               scoring= None ,
                                                               n_jobs=4,iid=False, cv=5)
gsearch8.fit(train_x, train_y)
gsearch8.grid_scores_, gsearch8.best_params_, gsearch8.best_score_
'''
selected subsample: 0.85
'''
#%%
'''
n_estimator = 350
max_depth: 5
min_samples_split: 2
min_samples_leaf: 20
max_features: 7
subsample: 0.85
'''
gbm_fit = GradientBoostingRegressor(n_estimators = 350, learning_rate=0.1, max_depth=5,
                          max_features=7, min_samples_leaf= 20, min_samples_split=2,
                          loss='huber', subsample = 0.85, random_state = 10)

gbm_fit.fit(train_x, train_y)

print 'R2:', gbm_fit.score(train_x, train_y), 'rmse:', rmse(train_y, gbm_fit.predict(train_x))
#%%
'''
halving learning rate and doubling number of trees
'''
gbm_fit_1 = GradientBoostingRegressor(n_estimators = 700, learning_rate=0.05, max_depth=5,
                          max_features=7, min_samples_leaf= 20, min_samples_split=2,
                          loss='huber', subsample = 0.85, random_state = 10)

gbm_fit_1.fit(train_x, train_y)

print 'R2:', gbm_fit_1.score(train_x, train_y), 'rmse:', rmse(train_y, gbm_fit_1.predict(train_x))
'''
It got better
'''
#%%
'''
1/10th learning rate and 10 number of trees
'''
gbm_fit_2 = GradientBoostingRegressor(n_estimators = 3500, learning_rate=0.01, max_depth=5,
                          max_features=7, min_samples_leaf= 20, min_samples_split=2,
                          loss='huber', subsample = 0.85, random_state = 10)

gbm_fit_2.fit(train_x, train_y)

print 'R2:', gbm_fit_2.score(train_x, train_y), 'rmse:', rmse(train_y, gbm_fit_2.predict(train_x))
'''
It got better again!! but not that much
'''
#%%
#Predicting test data
gbm_submission = np.exp(gbm_fit_2.predict(test_x))
final_result = pd.DataFrame(gbm_submission, index = test_df_munged.index, columns = ['SalePrice'])
final_result.to_csv('final_result3.csv')
#%%
from matplotlib import pyplot
#%matplotlib qt
pyplot.bar(range(len(gbm_fit_2.feature_importances_)), gbm_fit_2.feature_importances_)
pyplot.show()
imp = pd.DataFrame()
imp['var'] = train_df_munged.columns
imp['imp'] = gbm_fit_2.feature_importances_
###############################################################################################
#%%
train_x = munge_train.as_matrix()
train_y = label_df.as_matrix()
test_x = munge_test.as_matrix()
#Baseline Model
gbm0 = GradientBoostingRegressor(random_state=10)
gbm0.fit(train_x, train_y)

print 'R2:', gbm0.score(train_x, train_y), 'rmse:', rmse(train_y, gbm0.predict(train_x))
pyplot.bar(range(len(gbm0.feature_importances_)), gbm0.feature_importances_)
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
param_test1 = {'n_estimators':range(20,500,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, max_depth=7, max_features='sqrt',
                                               min_samples_leaf=15, min_samples_split=8, loss='huber', subsample = 0.8, random_state = 10),
                                                               param_grid = param_test1,
                                                               scoring= None ,
                                                               n_jobs=4,iid=False, cv=5)
gsearch1.fit(train_x, train_y)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
#{'n_estimators': 110}
#%%
param_test2 = {'max_depth':range(5,16,2), 'min_samples_split':range(5,16,2)}
gsearch2 = GridSearchCV(estimator = GradientBoostingRegressor(n_estimators = 110, learning_rate=0.1, max_features='sqrt',
                                               min_samples_leaf=15, loss='huber', subsample = 0.8, random_state = 10),
                                                               param_grid = param_test2,
                                                               scoring= None ,
                                                               n_jobs=4,iid=False, cv=5)
gsearch2.fit(train_x, train_y)
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_
#{'max_depth': 15, 'min_samples_split': 5**}
#%%
param_test3 = {'min_samples_split':range(1,6,1)}
gsearch3 = GridSearchCV(estimator = GradientBoostingRegressor(n_estimators = 110, learning_rate=0.1,max_depth = 15, max_features='sqrt',
                                               min_samples_leaf=15, loss='huber', subsample = 0.8, random_state = 10),
                                                               param_grid = param_test3,
                                                               scoring= None ,
                                                               n_jobs=4,iid=False, cv=5)
gsearch3.fit(train_x, train_y)
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
#{'min_samples_split': 1}
#%%
param_test4 = {'min_samples_leaf':range(30,71,10)}
gsearch4 = GridSearchCV(estimator = GradientBoostingRegressor(n_estimators = 110, learning_rate=0.1, max_depth= 15, max_features='sqrt',
                                               min_samples_split=1, loss='huber', subsample = 0.8, random_state = 10),
                                                               param_grid = param_test4,
                                                               scoring= None ,
                                                               n_jobs=4,iid=False, cv=5)
gsearch4.fit(train_x, train_y)
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_
#%%
param_test5 = {'min_samples_leaf':range(5,31,5)}
gsearch5 = GridSearchCV(estimator = GradientBoostingRegressor(n_estimators = 110, learning_rate=0.1, max_depth= 15, max_features='sqrt',
                                               min_samples_split=1, loss='huber', subsample = 0.8, random_state = 10),
                                                               param_grid = param_test5,
                                                               scoring= None ,
                                                               n_jobs=4,iid=False, cv=5)
gsearch5.fit(train_x, train_y)
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_
#{'min_samples_leaf': 20}
#%%
param_test6 = {'max_features':range(4,8,1)}
gsearch6 = GridSearchCV(estimator = GradientBoostingRegressor(n_estimators = 110, learning_rate=0.1, max_depth= 15,
                                               min_samples_leaf = 20, min_samples_split=1, loss='huber', subsample = 0.8, random_state = 10),
                                                               param_grid = param_test6,
                                                               scoring= None ,
                                                               n_jobs=4,iid=False, cv=5)
gsearch6.fit(train_x, train_y)
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_
#{'max_features': 6}
#%%
param_test7 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
gsearch7 = GridSearchCV(estimator = GradientBoostingRegressor(n_estimators = 110, learning_rate=0.1, max_depth= 15, max_features=6,
                                               min_samples_leaf = 20, min_samples_split=1, loss='huber', random_state = 10),
                                                               param_grid = param_test7,
                                                               scoring= None ,
                                                               n_jobs=4,iid=False, cv=5)
gsearch7.fit(train_x, train_y)
gsearch7.grid_scores_, gsearch7.best_params_, gsearch7.best_score_
#{'subsample': 0.8}
#%%
grad_boost = GradientBoostingRegressor(n_estimators = 110, learning_rate=0.1, max_depth= 15, max_features=6,
                                               min_samples_leaf = 20, min_samples_split=1, loss='huber',
                                               random_state = 10, subsample = 0.8)
grad_boost.fit(train_x, train_y)

print 'R2:', grad_boost.score(train_x, train_y), 'rmse:', rmse(train_y, grad_boost.predict(train_x))
#%%
'''
halving learning rate and doubling number of trees
'''
grad_boost_1 = GradientBoostingRegressor(n_estimators = 220, learning_rate=0.05, max_depth= 15, max_features=6,
                                               min_samples_leaf = 20, min_samples_split=1, loss='huber',
                                               random_state = 10, subsample = 0.8)
grad_boost_1.fit(train_x, train_y)

print 'R2:', grad_boost_1.score(train_x, train_y), 'rmse:', rmse(train_y, grad_boost_1.predict(train_x))
#%%
'''
1/10th learning rate and 10 number of trees
'''
grad_boost_2 = GradientBoostingRegressor(n_estimators = 1100, learning_rate=0.01, max_depth= 15, max_features=6,
                                               min_samples_leaf = 20, min_samples_split=1, loss='huber',
                                               random_state = 10, subsample = 0.8)
grad_boost_2.fit(train_x, train_y)

print 'R2:', grad_boost_2.score(train_x, train_y), 'rmse:', rmse(train_y, grad_boost_2.predict(train_x))
#%%
'''
1/20th learning rate and 20 number of trees
'''
grad_boost_3 = GradientBoostingRegressor(n_estimators = 2200, learning_rate=0.005, max_depth= 15, max_features=6,
                                               min_samples_leaf = 20, min_samples_split=1, loss='huber',
                                               random_state = 10, subsample = 0.8)
grad_boost_3.fit(train_x, train_y)

print 'R2:', grad_boost_3.score(train_x, train_y), 'rmse:', rmse(train_y, grad_boost_3.predict(train_x))
#%%
#Predicting test data
gbm_submission = np.exp(grad_boost_3.predict(test_x))
final_result = pd.DataFrame(gbm_submission, index = test_df_munged.index, columns = ['SalePrice'])
final_result.to_csv('final_result3.csv')