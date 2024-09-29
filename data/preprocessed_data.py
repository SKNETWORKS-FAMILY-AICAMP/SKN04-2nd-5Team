import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def convert_object_into_integer(df: pd.DataFrame):
    label_encoders = {}
    for column in df.columns:
        if df.dtypes[column] == object:
            label_encoder = LabelEncoder()
            df[column] = label_encoder.fit_transform(df[column])
            label_encoders.update({column: label_encoder})
    
    return df, label_encoders

def fillna_median(data: pd.DataFrame):
    data.AgeHH1 = data.AgeHH1.fillna(data.AgeHH1.median())
    data.AgeHH2 = data.AgeHH2.fillna(data.AgeHH2.median())
    data.PercChangeMinutes = data.PercChangeMinutes.fillna(data.PercChangeMinutes.median())
    data.PercChangeRevenues = data.PercChangeRevenues.fillna(data.PercChangeRevenues.median())
    data.MonthlyRevenue = data.MonthlyRevenue.fillna(data.MonthlyRevenue.median())
    data.MonthlyMinutes = data.MonthlyMinutes.fillna(data.MonthlyMinutes.median())
    data.TotalRecurringCharge = data.TotalRecurringCharge.fillna(data.TotalRecurringCharge.median())
    data.DirectorAssistedCalls = data.DirectorAssistedCalls.fillna(data.DirectorAssistedCalls.median())
    data.OverageMinutes = data.OverageMinutes.fillna(data.OverageMinutes.median())
    data.RoamingCalls = data.RoamingCalls.fillna(data.RoamingCalls.median())
    data.CurrentEquipmentDays = data.CurrentEquipmentDays.fillna(data.CurrentEquipmentDays.median())
    data.HandsetModels = data.HandsetModels.fillna(data.HandsetModels.median())
    data.Handsets = data.Handsets.fillna(data.Handsets.median())
    return data

def preprocessed_data(df: pd.DataFrame, is_fillna=False, drop_columns=['CustomerID', 'ServiceArea']):
    """
    ## 데이터 전처리 함수(원본유지)\n
    결측치 처리\n
    학습에 불필요한 컬럼 제거\n
    objec형 정수로 변환\n
    """
    data = df.copy()
    if not is_fillna:
        data = data.dropna()
    else:
        data = fillna_median(data)

    data = data.drop(columns=drop_columns)
    data, _ = convert_object_into_integer(data)

    return data