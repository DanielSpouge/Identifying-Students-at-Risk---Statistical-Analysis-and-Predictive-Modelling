import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def apply_feature_engineering(df_input, model_type="performance", include_g1=True):
    
    data = df_input.copy()
    
    data["Pedu_max"] = data[["Medu", "Fedu"]].max(axis=1)
    data["alc_avg"] = data[["Dalc", "Walc"]].mean(axis=1)
    data["absences_log"] = np.log1p(data["absences"])
    
    to_drop = ["Medu", "Fedu", "Dalc", "Walc", "absences", "G2"]
    
    if model_type == "risk":
        to_drop.extend(["absences_log", "G3"])
        
    if not include_g1:
        to_drop.append("G1")
        
    data.drop(columns=to_drop, inplace=True, errors="ignore")
    
    return data

def apply_formatting(df_input, scaler=None):
    data = df_input.copy()
    
    binary_cols = [col for col in data.select_dtypes(include=["object"]).columns 
                       if data[col].nunique() == 2]
    
    for col in binary_cols:
        
        data[col] = data[col].apply(lambda x: 1 if x in ["yes", "MS", "U", "T", "LE3", "M"] else 0)
        

    cat_cols = [col for col in data.select_dtypes(include=["object"]).columns 
                        if data[col].nunique() > 2]
    
    data = pd.get_dummies(data, columns=cat_cols, drop_first=True, dtype=int)
    
    cols_to_scale = [col for col in data.select_dtypes(include=[np.number]).columns
        if data[col].nunique() > 2 and col != "G3"]
    
    if scaler is None:
        scaler = MinMaxScaler()
        data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])
        return data, scaler
    else:
        data[cols_to_scale] = scaler.transform(data[cols_to_scale])
        return data
    
def full_preprocessing_pipeline(raw_df, model_type="performance", include_g1=True, scaler=None):
    
    df_engineered = apply_feature_engineering(raw_df, model_type=model_type, include_g1=include_g1)
    
    return apply_formatting(df_engineered, scaler=scaler)
