import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder

def rmv_outliers(df: pd.DataFrame, col: str, thresh: int) -> pd.DataFrame:
    
    return df.loc[df[col] < thresh].reset_index(drop=True)

def feat_eng_passenger_id(df: pd.DataFrame) -> pd.DataFrame:

    # Passenger ID can be split into two columns: Group and ID
    ## This is rather inefficient:
    # df['ID'] = df['PassengerId'].apply(lambda x: str(x).split('_')[1])
    # df['Group'] = df['PassengerId'].apply(lambda x: str(x).split('_')[0])
    # More efficient
    df = pd.concat([df, df['PassengerId'].str.split('_', expand=True)], axis=1)
    df.rename(columns={0: 'Group', 1: 'ID'}, inplace=True)

    # Dict to map the number of passengers in each group
    groups = df.groupby('Group')['ID'].count().to_dict()
    # Map the number of passengers in each group to the dataframe
    df['GroupSize'] = df['Group'].map(groups)

    # drop the Group and ID columns, as they are no longer needed since we derived the groupsize from them
    df.drop('Group', axis=1, inplace=True)
    df.drop('ID', axis=1, inplace=True)
    # drop the PassengerId column, as it is no longer needed either
    df.drop('PassengerId', axis=1, inplace=True)

    return df

def feat_eng_cabin(df: pd.DataFrame) -> pd.DataFrame:

    # Split up the Cabin feature as there are 3 sub features in it
    df[['Deck','CabinNo','CabinSide']] = df['Cabin'].str.split('/', expand=True)
    df['CabinNo'] = df['CabinNo'].astype(float)

    # CabinSide is binary with S and P, so we can map it to 0 and 1
    df['CabinSide'] = df['CabinSide'].map({'S': 0, 'P': 1})

    df.drop('Cabin', axis=1, inplace=True)

    return df

def imp_missing(df: pd.DataFrame) -> pd.DataFrame:

    # Impute missing values using KNNImputer. We could do column specific imputation, but this is a fair approach too
    df_float = df.select_dtypes(['float64','int64'])
    df_object = df.select_dtypes('object')

    imp = KNNImputer(n_neighbors=5)
    imp_data = imp.fit_transform(df_float)
    df_float = pd.DataFrame(data=imp_data, columns=df_float.columns)

    df[df_float.columns] = df_float

    ##use simple mode imputation for categorical variables
    # Destination does not matter much as seen above, so mode imputation is alright
    df['Destination'] = df['Destination'].fillna(value=df['Destination'].mode()[0])
    # Regarding Home Planet, mode imputation could skew the results, but it's a fair start
    df['HomePlanet'] = df['HomePlanet'].fillna(value=df['HomePlanet'].mode()[0])
    # Most people are traveling in Deck F or G, so mode imputation is reasonable
    df['Deck'] = df['Deck'].fillna(value=df['Deck'].mode()[0])

    return df

def encoding(df: pd.DataFrame) -> pd.DataFrame:

    ohe = OneHotEncoder(sparse=False)
    ohe_data = ohe.fit_transform(df.select_dtypes(include='object'))
    df_object = pd.DataFrame(data=ohe_data, columns=ohe.get_feature_names_out(df.select_dtypes(include='object').columns))

    df.drop(df.select_dtypes(include='object').columns, axis=1, inplace=True)
    df = pd.concat([df, df_object], axis=1)

    return df
