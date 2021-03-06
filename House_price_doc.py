# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 21:36:14 2017

@author: Nemish
"""

#House prices sample
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
#%matplotlib qt
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

#%%
#Outliers
Outliers_index = []
Outliers_index += list(House_price_train.sort_values(by = 'LotFrontage', ascending = False)[:2].index)
Outliers_index += list(House_price_train.sort_values(by = 'LotArea', ascending = False)[:4].index)
Outliers_index += list(House_price_train.sort_values(by = 'SalePrice', ascending = False)[:2].index)
House_price_train.drop([1299, 935, 314, 336, 250, 707, 692, 1183], inplace = True) # to ensure consistency
# derived variable = summ_livBsSF done later
#%%
def check_null(Dataframe):
    '''
    Returns columns with number of nulls
    '''
    a = Dataframe.isnull().sum()[Dataframe.isnull().sum()>0]
    if len(a) == 0:
        print 'Null'
    else:
        return a

def no_null_objects(data, columns=None):
    """
    Returns rows with no NaNs
    """
    if columns is None:
        columns = data.columns
    return data[np.logical_not(np.any(data[columns].isnull().values, axis=1))]
    
House_price_train = no_null_objects(House_price_train, columns = ['Electrical', 'RoofMatl'])
#%%
#Numerical Variables
House_price_train['summ_BsmtSF'] = House_price_train['BsmtUnfSF'] +\
                                    House_price_train['BsmtFinSF2'] + \
                                    House_price_train['BsmtFinSF1']
                                    
House_price_train['summ_Bathrooms'] = House_price_train['BsmtHalfBath'] +\
                                    House_price_train['BsmtFullBath']+\
                                    House_price_train['FullBath']+\
                                    House_price_train['HalfBath'] 
                                    
House_price_train['summ_livBsSF'] = House_price_train['GrLivArea'] + House_price_train['summ_BsmtSF']
                                  
num_var = ['LotFrontage', 'LotArea', 'MasVnrArea',
           'BedroomAbvGr', 'summ_Bathrooms', 'summ_livBsSF',
           'GarageCars', 'SalePrice']
           
train_num = House_price_train[num_var]
check_null(train_num)
#filling na with mean 
train_num = train_num.fillna(train_num.mean())
#%%
#Outlier Removal contd..
Outliers_index = list(House_price_train.sort_values(by = 'summ_livBsSF', ascending = False)[:2].index)
House_price_train.drop([524, 497], inplace = True) # to ensure consistency
train_num.drop([524, 497], inplace = True) # to ensure consistency

#%%
train_data = pd.DataFrame()

train_data['SalePrice'] = House_price_train['SalePrice']

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

House_price_train['summ_Condition'] = (House_price_train["mod_Condition1"] + House_price_train["mod_Condition2"])

train_data['summ_Condition'] = House_price_train['summ_Condition'].replace([0,1], 0)\
.replace([2,3], 1)\
.replace([4,5], 2)\
.replace([5,6,7,8,9], 3)

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

train_data["mod_HouseStyle"] = House_price_train["HouseStyle"].replace('1.5Unf', 0)\
.replace(['1.5Fin', 'SFoyer', '2.5Unf'], 1)\
.replace(['1Story', 'SLvl'], 2)\
.replace(['2Story', '2.5Fin'], 3)

train_data["mod_BsmtQual"] = House_price_train["BsmtQual"].replace(np.nan, 0)\
.replace(['Fa', 'TA'], 1)\
.replace('Gd', 2)\
.replace('Ex', 3)

train_data["mod_BsmtCond"] = House_price_train["BsmtCond"].replace(np.nan, 0)\
.replace(['Po'], 1)\
.replace('Fa', 2)\
.replace(['TA', 'Gd'], 3)

#train_data["mod_GarageType"] = House_price_train["GarageType"].replace(np.nan, 0)\
#.replace(['CarPort'], 1)\
#.replace('Detchd', 2)\
#.replace(['Basment', '2Types'], 3)\
#.replace(['Attchd'], 3)\
#.replace(['BuiltIn'], 4)

train_data["mod_SaleType"] = House_price_train["SaleType"].replace(['Oth', 'ConLI'] , 0)\
.replace(['COD', 'ConLD', 'ConLw'], 1)\
.replace('WD', 2)\
.replace(['CWD'], 3)\
.replace(['New'], 3)\
.replace(['Con'], 4)

train_data["mod_BldgType"] = House_price_train["BldgType"].replace(['2fmCon'] , 0)\
.replace(['Duplex', 'Twnhs'], 1)\
.replace('1Fam', 2)\
.replace(['TwnhsE'], 3)

train_data["mod_Foundation"] = House_price_train["Foundation"].replace(["BrkTil", "CBlock", "Slab", "Stone", "Wood"] , 0)\
.replace(['PConc'], 1)

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

train_data["mod_Electrical"] = House_price_train["Electrical"].replace(["Mix"] , 0)\
.replace(['FuseP'], 1)\
.replace(['FuseF', "FuseA"], 2)\
.replace(['SBrkr'], 3)

train_data["mod_FireplaceQu"] = House_price_train["FireplaceQu"].fillna("None")\
.map({"Ex": 3, "Gd": 3, "TA": 2, "Fa": 1, "Po": 1, "None": 0})

train_data["mod_PoolQC"] = House_price_train["PoolQC"].fillna("None")\
.map({"Ex": 2, "Gd": 1, "TA": 1, "Fa": 1, "Po": 1, "None": 0})

train_data["mod_SaleCondition"] = House_price_train["SaleCondition"].fillna("None")\
.map({"Partial": 2, "Abnorml": 1, "Family": 1, "Alloca": 1, "Normal": 1, "AdjLand": 0, "None": 0})

train_data["mod_Alley"] = House_price_train["Alley"].fillna("None")\
.map({"Pave": 2, "Grvl": 1, "None":0})

#train_data["mod_ExterQual"] = House_price_train["ExterQual"].fillna("None")\
#.map({"Ex": 3, "Gd": 2, "TA": 1, "Fa": 0, "Po": 0})

train_data["mod_CentralAir"] = House_price_train["CentralAir"].fillna("None")\
.map({"Y": 1, "N": 0})

#train_data["mod_KitchenQual"] = House_price_train["KitchenQual"].fillna("None")\
#.map({"Ex": 3, "Gd": 2, "TA": 1, "Fa": 0, "Po": 0})

train_data["mod_PavedDrive"] = House_price_train["PavedDrive"].fillna("None")\
.map({"Y":2, "P": 1, "N": 0})

train_data["mod_GarageFinish"] = House_price_train["GarageFinish"].fillna("None")\
.map({"Fin": 3, "RFn":2, "Unf": 1, "None": 0})

train_data["mod_OverallQual"] = (House_price_train["OverallQual"]-1)

train_data["mod_MSZoning"] = House_price_train["MSZoning"].fillna("None")\
.map({"FV":3, "RL": 2, "RH":1, "RM": 1, "C (all)": 0})

#train_data["mod_Exterior1st"] = House_price_train["Exterior1st"].fillna("None")\
#.map({"ImStucc":7, "Stone":7, "CemntBd":6, "VinylSd":5, "Plywood":4,\
#    "BrkFace":4, "HdBoard":3, "Stucco":3, "MetalSd":2, "Wd Sdng":2,\
#    "WdShing":2, "AsbShng": 1, "CBlock":1, "AsphShn": 1, "BrkComm": 0})
    
train_data["mod_MasVnrType"] = House_price_train["MasVnrType"].fillna("None")\
.map({"Stone": 2, "BrkFace":1, "BrkCmn": 0, "None": 0})

train_data["mod_RoofMatl"] = House_price_train["RoofMatl"].fillna("None")\
.map({"WdShngl": 1,"CompShg":0, "Metal":0, "WdShake":0, "Membran":0,\
      "Tar&Grv":0, "Roll":0, "ClyTile":0})
      
train_data["mod_ExterCond"] = House_price_train["ExterCond"].fillna("None")\
.map({"Ex":1, "Gd":1, "TA":1, "Fa":0,\
      "Po":0})

train_data["mod_BsmtExposure"] = House_price_train["BsmtExposure"].fillna("None")\
.map({"Av":2, "Gd":2, "Mn":2, "No":1,\
      "None":0})

train_data["mod_HeatingQC"] = House_price_train["HeatingQC"].fillna("None")\
.map({"Ex":1, "Gd":0, "TA":0, "Fa":0,\
      "Po":0})

train_data["mod_MSSubClass"] = House_price_train["MSSubClass"].replace([20,  70,  50, 190, 150,  45,  90, 120,  85,  80, 160,  75,
       180,  40] , 1)\
.replace([30], 0)\
.replace([60], 2)

train_data['mod_OverallCond'] =House_price_train['OverallCond'].apply(lambda x: 1 if x >= 5 else 0)

House_price_train['mod_OpenPorchSF'] =House_price_train['OpenPorchSF'].apply(lambda x: 0.2 if x != 0 else 0)
House_price_train['mod_EnclosedPorch'] =House_price_train['EnclosedPorch'].apply(lambda x: 0.3 if x != 0 else 0)
House_price_train['mod_3SsnPorch'] =House_price_train['3SsnPorch'].apply(lambda x: 0.7 if x != 0 else 0)
House_price_train['mod_ScreenPorch'] =House_price_train['ScreenPorch'].apply(lambda x: 0.11 if x != 0 else 0)

House_price_train['summ_porch_cond'] = House_price_train['mod_OpenPorchSF'] + \
     House_price_train['mod_EnclosedPorch'] + \
     House_price_train['mod_3SsnPorch'] + \
     House_price_train['mod_ScreenPorch']

train_data["mod_summ_porch_cond"] =House_price_train['summ_porch_cond'].apply(lambda x: 1 if x in (.2, .31) else 0)

train_data['mod_Fireplaces'] =House_price_train['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

train_data['mod_LowQualFinSF'] =House_price_train['LowQualFinSF'].apply(lambda x: 1 if x > 0 else 0)

def year_era(x):
    if x < 1950:
        return 0
    elif x < 2000:
        return 1
    else:
        return 2
House_price_train['era_YearBuilt'] = House_price_train['YearBuilt'].apply(year_era)
House_price_train['era_YearRemodAdd'] = House_price_train['YearRemodAdd'].apply(year_era)
House_price_train['mod_remod'] = np.where(House_price_train['era_YearBuilt'] == House_price_train['era_YearRemodAdd'], 0, 1)

train_data["mod_remod"] = House_price_train['mod_remod']
train_data['era_YearBuilt'] = House_price_train['era_YearBuilt']

check_null(train_data) #expected Null
#%%
#corr Heatmap
corrmat = train_data.corr().round(2)

corrmat_1 = pd.DataFrame()
for i in corrmat.columns:
    corrmat_1[i] = corrmat[i].apply(lambda x: 0 if abs(x) <= 0.5 else x)

sns.heatmap(corrmat_1, annot=True)
plt.xticks(rotation=30, ha = 'right')
plt.yticks(rotation=0)
plt.title('**Correlation Heatmap for features, 0 means corr less than 0.5**', weight = 'bold')
#plt.tick_params(axis  = 'both', color = 'Black')
#%%
def dummy_var(train_data, exception = []):
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
    
train_cat = dummy_var(train_data, exception = ['SalePrice'])

#%%
corrmat = train_num.corr().round(2)
sns.heatmap(corrmat, annot=True)
plt.xticks(rotation=30, ha = 'right')
plt.yticks(rotation=0)
#%%
#Linearity
#scatterplot
sns.set()
sns.pairplot(train_num, vars = list(train_num.columns[0:3]) +['SalePrice'])
plt.figure()
sns.pairplot(train_num, vars = list(train_num.columns[3: train_num.shape[1] + 1]))
plt.xticks(rotation=30, ha = 'right')
plt.yticks(rotation=0)
#%%
from scipy.stats import norm
from scipy import stats
def prob_plt(i, logcurve):
    if(logcurve):
        plt.figure()
        sns.distplot(np.log(train_num[train_num.columns[i]] + 1), fit=norm);
        plt.figure()
        stats.probplot(np.log(train_num[train_num.columns[i]] + 1), plot=plt)
    else:
        plt.figure()
        sns.distplot(train_num[train_num.columns[i]], fit=norm);
        plt.figure()
        stats.probplot(train_num[train_num.columns[i]], plot=plt)
        
i = 4
logcurve = 1
prob_plt(i, logcurve)
#%%
#Normal distribution
'''
log -> LotArea, MasVnrArea (n+1), summ_livBsSF, SalePrice
nolog -> LotFrontage, BedroomAbvGr, summ_Bathrooms, GarageCars
'''
train_num_1 = train_num[['LotFrontage', 'BedroomAbvGr', 'summ_Bathrooms', 'GarageCars']]
train_num_1['LotArea'] = train_num['LotArea'].apply(lambda x: np.log(x))
train_num_1['MasVnrArea'] = train_num['MasVnrArea'].apply(lambda x: np.log(x+1))
train_num_1['summ_livBsSF'] = train_num['summ_livBsSF'].apply(lambda x: np.log(x))
train_num_1['SalePrice'] = train_num['SalePrice'].apply(lambda x: np.log(x))
#%%
#Multicollinearity check with VIF
corrmat = train_num_1.corr()
def vif_func(x):
    try:
        return 1.0/(1-x**2)
    except ZeroDivisionError:
        return -1
Vif = corrmat.applymap(vif_func)
sns.heatmap(Vif, annot=True)
plt.xticks(rotation=30, ha = 'right')
plt.yticks(rotation=0)
#Check if VIF>10, for multi-collinearity

#%%

def Autocorr_heteroSked(train_num_1):
    '''
    Auto-Correlation  1.5 < d < 2.5 and Heterosedascity: Goldfeld-Quandt test as near to 1
    NOTE: y variable is set to SalePrice
    '''
    #Auto-Correlation  1.5 < d < 2.5 and Heterosedascity: Goldfeld-Quandt test
    from statsmodels.stats.stattools import durbin_watson
    #from statsmodels.stats.stattools import het_goldfeldquandt
    from sklearn.linear_model import LinearRegression
    dw = {}
    F = {}
    for i in train_num_1:
        y = train_num_1[['SalePrice']]
        X = train_num_1[[i]]
        #mod = ols(y = train_num_1[['SalePrice']].as_matrix(), x = train_num_1[i].as_matrix())
        lr = LinearRegression()
        lr.fit(X, y)
        resid = y - lr.predict(X)
        resd = pd.DataFrame(resid.values, columns = ['residual'])
        resd['X'] = X.values
        resd['y'] = y.values
        reshd = resd.sort_values(by = 'X', ascending = True)
        
        #Autocorrelation: durbin watson
        dw[i] = durbin_watson(reshd['residual'])
        dw_res = dict((k,v) for k, v in dw.items() if v > 2.5 or v < 1.5)
        
        #Heteroscedascity: Golfeld-Quant test
        from math import ceil
        N1 = reshd.residual[0:(len(reshd)/2 - int(ceil(len(reshd)/100)))]**2
        N2 = reshd.residual[(len(reshd)/2 + int(ceil(len(reshd)/100))) : len(reshd)]**2
        F[i] = (sum(N2)/len(N2)) / (sum(N1)/len(N1))
        
        plt.figure()
        ax = sns.regplot(x = X, y = 'residual', data = reshd, label = i)
        ax.legend(loc="best")
    tup = (('durbin_watson', dw), ('dw_res', dw_res), ('Golfeld_Quant', F))
    return tup

a = Autocorr_heteroSked(train_num_1)    
'''
# The slope of residual regression line is 0 and F value is also close to 1, so homoskedascity
# if F value is less than 1 means, initial values have more variance and viceversa
# dw also checks out as dw_res is null (except SalePrice which is OK)
'''
#%%
#Scaling numerical data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(train_num_1.iloc[:,:-1])
train_num_2 = pd.DataFrame(scaler.transform(train_num_1.iloc[:,:-1]), columns = train_num_1.iloc[:,:-1].columns, index = train_num_1.iloc[:,:-1].index)

target = train_num_1.SalePrice
#join data
train_join = train_cat.join(train_num_2)
#%%
#Linear Regression
from sklearn.linear_model import LinearRegression
regr = LinearRegression()

x_matrix = train_join.as_matrix()
y_matrix = target.as_matrix()

regr.fit(x_matrix, y_matrix)
#mean squared error
SSE = sum((np.exp(regr.predict(x_matrix)) - np.exp(y_matrix))**2)
SST = sum((np.exp(y_matrix) - np.exp(np.mean(y_matrix)))**2)
R2 = 1-SSE/SST 
n = len(x_matrix)
m = x_matrix.shape[1]
R2adj = 1 - (1- R2)*(n-1)/(n-m)
#http://stats.stackexchange.com/questions/32596/what-is-the-difference-between-coefficient-of-determination-and-mean-squared
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets)**2).mean())
rmse(regr.predict(x_matrix), y_matrix)

from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(np.exp(y_matrix), regr.predict(x_matrix)))
#%%
Residual_y = (regr.predict(x_matrix) - y_matrix)
plt.hist(Residual_y, bins = np.arange(min(Residual_y), max(Residual_y) + .01, .01))

plt.stem(y_matrix, Residual_y, markerfmt = 'ro', linefmt = 'g--', basefmt = 'm:')
plt.title('Residual v/s SalePrice')
plt.ylabel('Residual')
plt.xlabel('SalePrice')
#%%
from sklearn.ensemble import GradientBoostingRegressor  #GBM algorithm
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

gbm_y = y_matrix - regr.predict(x_matrix)
gbm_x = x_matrix
param_test1 = {'n_estimators':range(20,81,10)}
#min_sample_split = 0.5-1% of 1459
gsearch1 = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, max_depth=3, max_features='sqrt',
                                               min_samples_leaf=15, min_samples_split=7, loss='huber', subsample = 0.8, random_state = 10),
                                                               param_grid = param_test1,
                                                               scoring= None ,
                                                               n_jobs=4,iid=False, cv=5)
gsearch1.fit(x_matrix, y_matrix)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
#%%
param_test2 = {'n_estimators':range(150,500,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, max_depth=3, max_features='sqrt',
                                               min_samples_leaf=15, min_samples_split=7, loss='huber', subsample = 0.8, random_state = 10),
                                                               param_grid = param_test2,
                                                               scoring= None ,
                                                               n_jobs=4,iid=False, cv=5)
gsearch1.fit(x_matrix, y_matrix)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
#%%
param_test2 = {'max_depth':range(5,16,2), 'min_samples_split':range(1,9,1)}

gsearch2 = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, n_estimators=430, max_features='sqrt', subsample=0.8, random_state=10), 
param_grid = param_test2, scoring= None,n_jobs=4,iid=False, cv=5)

gsearch2.fit(x_matrix, y_matrix)
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_
#%%
param_test3 = {'min_samples_leaf':range(1,9,1), 'min_samples_split':range(9,15,1)}

gsearch3 = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, n_estimators=430,
                                                              max_depth = 9,
                                                              max_features='sqrt', subsample=0.8,
                                                              random_state=10), 
param_grid = param_test3, scoring= None,n_jobs=4,iid=False, cv=5)

gsearch3.fit(x_matrix, y_matrix)
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
#%%
param_test4 = {'max_features':range(5,14,2)}

gsearch4 = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, n_estimators=430, min_samples_leaf =3,
                                                              min_samples_split = 9, max_depth = 9,
                                                              subsample=0.8,
                                                              random_state=10, loss='huber'), 
param_grid = param_test4, scoring= None,n_jobs=4,iid=False, cv=5)

gsearch4.fit(x_matrix, y_matrix)
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_
#%%
param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}

gsearch5 = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, n_estimators=430, min_samples_leaf =4,
                                                              min_samples_split = 3, max_depth = 9,
                                                              max_features = 11,
                                                              random_state=10, loss='huber'), 
param_grid = param_test5, scoring= None,n_jobs=4,iid=False, cv=5)

gsearch5.fit(x_matrix, y_matrix)
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_
#%%

gbm_fit = GradientBoostingRegressor(learning_rate=0.1, n_estimators=430, min_samples_leaf =4,
                                                              min_samples_split = 3, max_depth = 9,
                                                              max_features = 9, subsample =0.85,
                                                              random_state=10, loss='huber')
gbm_fit.fit(x_matrix, y_matrix)
gbm_ans = gbm_fit.predict(x_matrix)

Residual_y = (gbm_ans - y_matrix)       
plt.hist(Residual_y, bins = np.arange(min(Residual_y), max(Residual_y) + .01, .01))    

plt.stem(y_matrix, Residual_y, markerfmt = 'ro', linefmt = 'g--', basefmt = 'm:')
plt.title('Residual v/s SalePrice')
plt.ylabel('Residual')
plt.xlabel('SalePrice')    

SSE = sum((np.exp(gbm_ans) - np.exp(y_matrix))**2)
SST = sum((np.exp(y_matrix) - np.exp(np.mean(y_matrix)))**2)
R2 = 1-SSE/SST 
n = len(x_matrix)
m = x_matrix.shape[1]
R2adj = 1 - (1- R2)*(n-1)/(n-m)
#http://stats.stackexchange.com/questions/32596/what-is-the-difference-between-coefficient-of-determination-and-mean-squared                                         

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets)**2).mean())
rmse(gbm_ans, y_matrix)