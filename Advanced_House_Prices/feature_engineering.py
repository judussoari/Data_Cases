import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder


def new_feats(df_train,df_test):
    df_merged = pd.concat([df_train,df_test], axis=0).reset_index(drop=True)
    df_merged.drop('SalePrice', axis=1, inplace=True)

    # Bathrooms
    df_merged['TotalBath'] = df_merged['BsmtFullBath'] + 0.5*df_merged['BsmtHalfBath'] + df_merged['FullBath'] + 0.5*df_merged['HalfBath']

    # Porch
    df_merged['TotalPorchSF'] = df_merged['WoodDeckSF'] +  df_merged['OpenPorchSF'] +  df_merged['EnclosedPorch'] +  df_merged['3SsnPorch'] + df_merged['ScreenPorch'] 

    # Dates sold
    df_merged['MoSold'] = df_merged['MoSold'].astype(int).astype(str)
    df_merged['YrSold'] = df_merged['YrSold'].astype(int).astype(str)

    # Binary Features of Pool, Fireplace etc exist
    df_merged['HasPool'] = (df_merged['PoolArea'] > 0).astype(int)
    df_merged['HasFireplace'] = (df_merged['Fireplaces'] > 0).astype(int)
    df_merged['HasGarage'] = (df_merged['GarageArea'] > 0).astype(int)
    df_merged['Has2ndFloor'] = (df_merged['2ndFlrSF'] > 0).astype(int)

    df_merged['MoSold'] = df_merged['MoSold'].astype(int)
    df_merged['YrSold'] = df_merged['YrSold'].astype(int)

    return df_merged

def drop_feats(df_merged):
    # Drop features that do not provide new info
    to_drop = []
    for col in df_merged.columns:
        most_occuring = df_merged[col].value_counts().iloc[0]
        if most_occuring/len(df_merged) > 0.95:
            to_drop.append(col)
    to_drop.append('GarageCars')
    to_drop.append('GarageYrBlt') # from correlation analysis
    df_merged.drop(to_drop, axis=1, inplace=True)

    return df_merged

def encode(df_merged):
    # Encodings
    ## Ordinal
    ordinal = OrdinalEncoder()
    ordinal_columns = [
        'LotShape','LandContour','ExterQual','ExterCond','HeatingQC','KitchenQual',
        'Functional','PavedDrive'
        ]
    ordinal_data = ordinal.fit_transform(df_merged[ordinal_columns])
    ordinal_df = pd.DataFrame(ordinal_data, columns=[ordinal_columns])
    df_merged.drop(ordinal_columns, axis=1, inplace=True)
    df_merged = pd.concat([df_merged, ordinal_df], axis=1)

    ## OneHot
    onehot = OneHotEncoder(drop='if_binary', sparse=False)
    onehot_cols = ['MSZoning','Alley','LotConfig','Neighborhood','Condition1','BldgType','HouseStyle',
                'RoofStyle','Exterior1st','Exterior2nd','MasVnrType','Foundation','BsmtQual','BsmtCond',
                'BsmtExposure','BsmtFinType1','BsmtFinType2','CentralAir','Electrical','FireplaceQu','GarageType','GarageFinish',
                'GarageQual','GarageCond','Fence','SaleType','SaleCondition']
    onehot_data = onehot.fit_transform(df_merged[onehot_cols])
    onehot_df = pd.DataFrame(onehot_data, columns=onehot.get_feature_names_out(onehot_cols))
    df_merged.drop(onehot_cols, axis=1, inplace=True)
    df_merged = pd.concat([df_merged,onehot_df], axis=1)

    return df_merged

def split_merged(df_merged, df_train):
    # split up again to train and test
    df_train_proc = df_merged.loc[:len(df_train)-1,:].reset_index(drop=True)
    df_test_proc = df_merged.loc[len(df_train):, :].reset_index(drop=True)

    df_train_proc['SalePrice'] = df_train['SalePrice'].copy()

    return df_train_proc, df_test_proc