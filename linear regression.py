# -*- coding: utf-8 -*-
"""
Created on Wed Jul 05 19:06:03 2017

@author: Nemish
"""

#House prices sample
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
label_df["SalePrice"] = np.log1p(House_price_train["SalePrice"])
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
from sklearn.linear_model import Lasso, Ridge, LassoCV
from sklearn.model_selection import cross_val_score

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, train_df_munged, label_df, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_lasso = [rmse_cv(Lasso(alpha = alpha)).mean() 
            for alpha in alphas]
cv_lasso = pd.Series(cv_lasso, index = alphas)
cv_lasso.plot(title = "Validation")
plt.xlabel("alpha")
plt.ylabel("rmse")

cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
            for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation")
plt.xlabel("alpha")
plt.ylabel("rmse")

cv_ridge.min()
cv_lasso.min()
#%%
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(train_df_munged, label_df)
rmse_cv(model_lasso).mean()
coef = pd.Series(model_lasso.coef_, index = train_df_munged.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
#%%
imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
#matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")

lasso_preds = np.expm1(model_lasso.predict(test_df_munged))
final_result = pd.DataFrame(lasso_preds, index = test_df_munged.index, columns = ['SalePrice'])
final_result.to_csv('final_result5.csv')
#%%
final_pred = .5*lasso_preds + 0.5*gbm_submission
final_result = pd.DataFrame(final_pred, index = test_df_munged.index, columns = ['SalePrice'])
final_result.to_csv('final_result6.csv')