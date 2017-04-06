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
    House_price_test = pd.read_csv('test.csv')
    House_price_test.index = House_price_test['Id']
    #no_use
    del House_price_test['Id']
    #Multicollinearity_cat
    del House_price_test['ExterQual']
    del House_price_test['GarageType']
    del House_price_test['KitchenQual']
    #Multicollinearity_num
    del House_price_test['GarageArea']
    del House_price_test['TotalBsmtSF']
    del House_price_test['TotRmsAbvGrd']
    del House_price_test['GarageYrBlt']
#    #No information gain
#    del House_price_train['YrSold']
#    del House_price_train['MoSold']
#    del House_price_train['MiscVal']
#    del House_price_train['PoolArea']
#    del House_price_train['WoodDeckSF']
#    del House_price_train['KitchenAbvGr']
    
    iitial+=1
#%%
check_null(House_price_test)
#list_of_vars = train_num_2.columns.append(train_data.columns)
#
#def multi_del(list1, texts):
#    '''
#    delete texts(list) from list1
#    '''
#    for i in texts:
#        list1 = [ s.replace(i, '') for s in list1]
#    return list1
#multi_del(list_of_vars, ['mod_', 'summ_', 'era_'])
##%%
#%%
test_data = pd.DataFrame()

House_price_test["mod_Condition1"] = House_price_test["Condition1"].replace('Artery', 0)\
.replace(['Feedr', 'RRAe'], 1)\
.replace(['Norm', 'RRAn'], 2)\
.replace(['RRNe', 'PosN'], 3)\
.replace(['PosA', 'RRNn'], 4)

House_price_test["mod_Condition2"] = House_price_test["Condition2"].replace('Artery', 0)\
.replace(['Feedr', 'RRAe'], 1)\
.replace(['Norm', 'RRAn'], 2)\
.replace(['RRNe', 'PosN'], 3)\
.replace(['PosA', 'RRNn'], 4)

House_price_test['summ_Condition'] = (House_price_test["mod_Condition1"] + House_price_test["mod_Condition2"])

test_data['summ_Condition'] = House_price_test['summ_Condition'].replace([0,1], 0)\
.replace([2,3], 1)\
.replace([4,5], 2)\
.replace([5,6,7,8,9], 3)

test_data["mod_Neighborhood"] = House_price_test["Neighborhood"].replace('MeadowV', 0)\
.replace(['IDOTRR', 'BrDale'], 1)\
.replace(['OldTown', 'Edwards', 'BrkSide'], 2)\
.replace(['Sawyer', 'Blueste', 'SWISU', 'NAmes', 'NPkVill', 'Mitchel'], 3)\
.replace(['SawyerW', 'Gilbert', 'NWAmes'], 4)\
.replace(['Blmngtn', 'CollgCr', 'ClearCr', 'Crawfor'], 5)\
.replace(['Veenker', 'Somerst', 'Timber'], 6)\
.replace(['StoneBr'], 7)\
.replace(['NoRidge'], 8)\
.replace(['NridgHt'], 9)

test_data["mod_HouseStyle"] = House_price_test["HouseStyle"].replace('1.5Unf', 0)\
.replace(['1.5Fin', 'SFoyer', '2.5Unf'], 1)\
.replace(['1Story', 'SLvl'], 2)\
.replace(['2Story', '2.5Fin'], 3)

test_data["mod_BsmtQual"] = House_price_test["BsmtQual"].replace(np.nan, 0)\
.replace(['Fa', 'TA'], 1)\
.replace('Gd', 2)\
.replace('Ex', 3)

test_data["mod_BsmtCond"] = House_price_test["BsmtCond"].replace(np.nan, 0)\
.replace(['Po'], 1)\
.replace('Fa', 2)\
.replace(['TA', 'Gd'], 3)

#test_data["mod_GarageType"] = House_price_test["GarageType"].replace(np.nan, 0)\
#.replace(['CarPort'], 1)\
#.replace('Detchd', 2)\
#.replace(['Basment', '2Types'], 3)\
#.replace(['Attchd'], 3)\
#.replace(['BuiltIn'], 4)

test_data["mod_SaleType"] = House_price_test["SaleType"].replace(['Oth', 'ConLI'] , 0)\
.replace(['COD', 'ConLD', 'ConLw'], 1)\
.replace('WD', 2)\
.replace(['CWD'], 3)\
.replace(['New'], 3)\
.replace(['Con'], 4)

test_data["mod_BldgType"] = House_price_test["BldgType"].replace(['2fmCon'] , 0)\
.replace(['Duplex', 'Twnhs'], 1)\
.replace('1Fam', 2)\
.replace(['TwnhsE'], 3)

test_data["mod_Foundation"] = House_price_test["Foundation"].replace(["BrkTil", "CBlock", "Slab", "Stone", "Wood"] , 0)\
.replace(['PConc'], 1)

House_price_test["mod_BsmtFinType1"] = House_price_test["BsmtFinType1"].replace(np.nan, 0)\
.replace(['LwQ'] , 1)\
.replace(['BLQ'], 2)\
.replace('Rec', 3)\
.replace(['ALQ'], 4)\
.replace(['Unf'], 5)\
.replace(['GLQ'], 6)

House_price_test["mod_BsmtFinType2"] = House_price_test["BsmtFinType2"].replace(np.nan, 0)\
.replace(['LwQ'] , 1)\
.replace(['BLQ'], 2)\
.replace('Rec', 3)\
.replace(['ALQ'], 4)\
.replace(['Unf'], 5)\
.replace(['GLQ'], 6)

test_data["summ_BsmtFinType"] = (House_price_test["mod_BsmtFinType1"] +\
 House_price_test["mod_BsmtFinType2"]).astype(int)
 
test_data["summ_BsmtFinType"] = test_data["summ_BsmtFinType"].replace(["1", "2", "3", "4", "5", "6", "7", "8", "9"] , 1)\
.replace(['0'], 0)\
.replace(['10'], 2)\
.replace(['11'], 3)

test_data["mod_Electrical"] = House_price_test["Electrical"].replace(["Mix"] , 0)\
.replace(['FuseP'], 1)\
.replace(['FuseF', "FuseA"], 2)\
.replace(['SBrkr'], 3)

test_data["mod_FireplaceQu"] = House_price_test["FireplaceQu"].fillna("None")\
.map({"Ex": 3, "Gd": 3, "TA": 2, "Fa": 1, "Po": 1, "None": 0})

test_data["mod_PoolQC"] = House_price_test["PoolQC"].fillna("None")\
.map({"Ex": 2, "Gd": 1, "TA": 1, "Fa": 1, "Po": 1, "None": 0})

test_data["mod_SaleCondition"] = House_price_test["SaleCondition"].fillna("None")\
.map({"Partial": 2, "Abnorml": 1, "Family": 1, "Alloca": 1, "Normal": 1, "AdjLand": 0, "None": 0})

test_data["mod_Alley"] = House_price_test["Alley"].fillna("None")\
.map({"Pave": 2, "Grvl": 1, "None":0})

#test_data["mod_ExterQual"] = House_price_test["ExterQual"].fillna("None")\
#.map({"Ex": 3, "Gd": 2, "TA": 1, "Fa": 0, "Po": 0})

test_data["mod_CentralAir"] = House_price_test["CentralAir"].fillna("None")\
.map({"Y": 1, "N": 0})

#test_data["mod_KitchenQual"] = House_price_test["KitchenQual"].fillna("None")\
#.map({"Ex": 3, "Gd": 2, "TA": 1, "Fa": 0, "Po": 0})

test_data["mod_PavedDrive"] = House_price_test["PavedDrive"].fillna("None")\
.map({"Y":2, "P": 1, "N": 0})

test_data["mod_GarageFinish"] = House_price_test["GarageFinish"].fillna("None")\
.map({"Fin": 3, "RFn":2, "Unf": 1, "None": 0})

test_data["mod_OverallQual"] = (House_price_test["OverallQual"]-1)

test_data["mod_MSZoning"] = House_price_test["MSZoning"].fillna("None")\
.map({"FV":3, "RL": 2, "RH":1, "RM": 1, "C (all)": 0})

#test_data["mod_Exterior1st"] = House_price_test["Exterior1st"].fillna("None")\
#.map({"ImStucc":7, "Stone":7, "CemntBd":6, "VinylSd":5, "Plywood":4,\
#    "BrkFace":4, "HdBoard":3, "Stucco":3, "MetalSd":2, "Wd Sdng":2,\
#    "WdShing":2, "AsbShng": 1, "CBlock":1, "AsphShn": 1, "BrkComm": 0})
    
test_data["mod_MasVnrType"] = House_price_test["MasVnrType"].fillna("None")\
.map({"Stone": 2, "BrkFace":1, "BrkCmn": 0, "None": 0})

test_data["mod_RoofMatl"] = House_price_test["RoofMatl"].fillna("None")\
.map({"WdShngl": 1,"CompShg":0, "Metal":0, "WdShake":0, "Membran":0,\
      "Tar&Grv":0, "Roll":0, "ClyTile":0})
      
test_data["mod_ExterCond"] = House_price_test["ExterCond"].fillna("None")\
.map({"Ex":1, "Gd":1, "TA":1, "Fa":0,\
      "Po":0})

test_data["mod_BsmtExposure"] = House_price_test["BsmtExposure"].fillna("None")\
.map({"Av":2, "Gd":2, "Mn":2, "No":1,\
      "None":0})

test_data["mod_HeatingQC"] = House_price_test["HeatingQC"].fillna("None")\
.map({"Ex":1, "Gd":0, "TA":0, "Fa":0,\
      "Po":0})

test_data["mod_MSSubClass"] = House_price_test["MSSubClass"].replace([20,  70,  50, 190, 150,  45,  90, 120,  85,  80, 160,  75,
       180,  40] , 1)\
.replace([30], 0)\
.replace([60], 2)

test_data['mod_OverallCond'] =House_price_test['OverallCond'].apply(lambda x: 1 if x >= 5 else 0)

House_price_test['mod_OpenPorchSF'] =House_price_test['OpenPorchSF'].apply(lambda x: 0.2 if x != 0 else 0)
House_price_test['mod_EnclosedPorch'] =House_price_test['EnclosedPorch'].apply(lambda x: 0.3 if x != 0 else 0)
House_price_test['mod_3SsnPorch'] =House_price_test['3SsnPorch'].apply(lambda x: 0.7 if x != 0 else 0)
House_price_test['mod_ScreenPorch'] =House_price_test['ScreenPorch'].apply(lambda x: 0.11 if x != 0 else 0)

House_price_test['summ_porch_cond'] = House_price_test['mod_OpenPorchSF'] + \
     House_price_test['mod_EnclosedPorch'] + \
     House_price_test['mod_3SsnPorch'] + \
     House_price_test['mod_ScreenPorch']

test_data["mod_summ_porch_cond"] =House_price_test['summ_porch_cond'].apply(lambda x: 1 if x in (.2, .31) else 0)

test_data['mod_Fireplaces'] =House_price_test['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

test_data['mod_LowQualFinSF'] =House_price_test['LowQualFinSF'].apply(lambda x: 1 if x > 0 else 0)

def year_era(x):
    if x < 1950:
        return 0
    elif x < 2000:
        return 1
    else:
        return 2
House_price_test['era_YearBuilt'] = House_price_test['YearBuilt'].apply(year_era)
House_price_test['era_YearRemodAdd'] = House_price_test['YearRemodAdd'].apply(year_era)
House_price_test['mod_remod'] = np.where(House_price_test['era_YearBuilt'] == House_price_test['era_YearRemodAdd'], 0, 1)

test_data["mod_remod"] = House_price_test['mod_remod']
test_data['era_YearBuilt'] = House_price_test['era_YearBuilt']

#fillna with 0
test_data = test_data.fillna(0)
check_null(test_data) #expected Null
#Assigning float values, so bulk change to int
test_data = test_data.astype(int)
#%%
#Dummy
test_cat = dummy_var(test_data)
#%%
#Numerical Variables
House_price_test['summ_BsmtSF'] = House_price_test['BsmtUnfSF'] +\
                                    House_price_test['BsmtFinSF2'] + \
                                    House_price_test['BsmtFinSF1']
                                    
House_price_test['summ_Bathrooms'] = House_price_test['BsmtHalfBath'] +\
                                    House_price_test['BsmtFullBath']+\
                                    House_price_test['FullBath']+\
                                    House_price_test['HalfBath'] 
                                    
House_price_test['summ_livBsSF'] = House_price_test['GrLivArea'] + House_price_test['summ_BsmtSF']
                                  
num_var = ['LotFrontage', 'LotArea', 'MasVnrArea',
           'BedroomAbvGr', 'summ_Bathrooms', 'summ_livBsSF',
           'GarageCars']
           
test_num = House_price_test[num_var]
check_null(test_num)
#filling na with mean 
test_num = test_num.fillna(test_num.mean())
#%%
#Normal distribution
'''
log -> LotArea, MasVnrArea (n+1), summ_livBsSF
nolog -> LotFrontage, BedroomAbvGr, summ_Bathrooms, GarageCars
'''
test_num_1 = test_num[['LotFrontage', 'BedroomAbvGr', 'summ_Bathrooms', 'GarageCars']]
test_num_1['LotArea'] = test_num['LotArea'].apply(lambda x: np.log(x))
test_num_1['MasVnrArea'] = test_num['MasVnrArea'].apply(lambda x: np.log(x+1))
test_num_1['summ_livBsSF'] = test_num['summ_livBsSF'].apply(lambda x: np.log(x))

#%%
#Scale
test_num_2 = pd.DataFrame(scaler.transform(test_num_1), columns = test_num_1.columns, index = test_num_1.index)
#Join
test_join = test_cat.join(test_num_2)
#%%
#ML Part
test_y = regr.predict(test_join.as_matrix()) #log values
final_result = pd.DataFrame(np.exp(test_y), index = test_join.index, columns = ['SalePrice'])
final_result.to_csv('final_result.csv')
