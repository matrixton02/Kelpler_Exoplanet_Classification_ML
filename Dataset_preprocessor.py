import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_dataset(filepath):                                                          # filters the data to extract our relevant features and also convert the outout to 1 or 0 instead of CONFIRMED OR NOT CONFORMED OR FALSE POSITIVE
    df=pd.read_csv(filepath,comment='#')
    feature_comlumns=[
        'koi_period',           #Orbital Period
        'koi_depth',            #Transit depth
        'koi_duration',         #Transit Duration
        'koi_prad',             #Planetary radius
        'koi_teq',              #Equilibrium temperature
        'koi_insol',            #Insolation flux
        'koi_steff',            #Stellar effective temperaure
        'koi_srad',             #Stellar radius
        'koi_smass'             #Stellar mass
    ]

    target_column='koi_disposition'

    df_filtered=df[df[target_column].isin(['CONFIRMED','FALSE POSITIVE'])].copy()

    df_filtered['target']=(df_filtered[target_column]=='CONFIRMED').astype(int)
    df_final=df_filtered[feature_comlumns+['target']].dropna()
    print(f"Dataset shape: {df_final.shape}")
    print(f"Features used: {feature_comlumns}")
    print(f"Class distribution:")
    print(df_final['target'].value_counts())
    print(f"Exoplanet percentage: {df_final['target'].mean()*100:.2f}%")

    X=df_final[feature_comlumns].values
    y=df_final['target'].values

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)                                                # we tranform the variable so that the model doesn't get diverted by very large values so we need to scale it all down but still maintaing the ratio
    X_test_scaled=scaler.transform(X_test)
    return X_train,X_test,y_train,y_test,X_train_scaled,X_test_scaled,feature_comlumns