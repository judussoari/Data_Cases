# Missing Values

##This is not a time series. So methods like imputation or rolling average do not work.

##Let's use KNN Imputation for the numerical columns

##For categorical columns: NaN values are actually valuable information (e.g. No Pool). Replace NaN with sth like 'No' 

##To do this, I need to merge train and test first, then seperate the datasets according to dtypes, then impute separately, then merge again


import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


def preprocess_missing(df_train, df_test):
    df_merged = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
    df_merged.drop('SalePrice', axis=1, inplace=True) # No Imputation of the target variable needed

    # Distinguish between numerical and categorical features
    df_merged_cont = df_merged.select_dtypes(include=['int64','float64'])
    df_merged_cat = df_merged.select_dtypes(include='object')

    # Continuous Variables
    imputer = KNNImputer(n_neighbors=5) 
    imputed_data = imputer.fit_transform(df_merged_cont)
    df_merged_cont = pd.DataFrame(imputed_data, columns=df_merged_cont.columns)

    # Categorical Variables
    # Change NaN to 'No' for this list of columns.
    replace_list = [
        'Alley','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
        'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PoolQC',
        'Fence','MiscFeature'
    ]
    df_merged_cat.loc[:,replace_list] = df_merged_cat.loc[:,replace_list].fillna(value='No')
    # For the rest of categorical columns, fill using mode imputation
    for col in df_merged_cat.columns:
        df_merged_cat[col].fillna(df_merged_cat[col].mode()[0], inplace=True) 

    # merge back categorical and numerical sub dataframes to one dataframe with no missing values
    df_merged = pd.concat([df_merged_cont, df_merged_cat], axis=1)

    # split up again to train and test
    df_train_proc = df_merged.loc[:len(df_train)-1,:].reset_index(drop=True)
    df_test_proc = df_merged.loc[len(df_train):, :].reset_index(drop=True)

    df_train_proc['SalePrice'] = df_train['SalePrice'].copy()

    return df_train_proc, df_test_proc

def drop_outliers(df_train):

    df_train = df_train[df_train['SalePrice'] < 700000]
    df_train = df_train[df_train['GrLivArea'] < 4000]
    df_train = df_train[df_train['LotArea'] < 150000]
    df_train = df_train[df_train['LotFrontage'] < 300]
    df_train = df_train[df_train['1stFlrSF'] < 4000]

    df_train = df_train.reset_index(drop=True)

    return df_train