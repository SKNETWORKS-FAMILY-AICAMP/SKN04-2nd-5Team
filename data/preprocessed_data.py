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

def preprocessed_data(df: pd.DataFrame):
    """
    ## 데이터 전처리 함수\n
    결측치 처리\n
    CustomerID 제거\n
    objec형 정수로 변환\n
    """
    data = df.dropna()
    # CustomerID는 학습과 관련이 없어 삭제하고 진행
    data = data.drop(columns='CustomerID')
    data, _ = convert_object_into_integer(data)

    return data