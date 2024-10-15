import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

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

def convert_continuous_to_categorical(data: pd.DataFrame):
    data['TotalRecurringCharge_label'] = pd.cut(data.TotalRecurringCharge, 8, labels=range(8)).astype(int)
    data['MonthsInService_label'] = pd.cut(data.MonthsInService, 6, labels=range(6)).astype(int)
    data['AgeHH1_label'] = pd.cut(data.AgeHH1, 5, labels=range(5)).astype(int)
    data['AgeHH2_label'] = pd.cut(data.AgeHH2, 5, labels=range(5)).astype(int)
    data['MonthlyMinutes_label'] = pd.cut(data.MonthlyMinutes, 10, labels=range(10)).astype(int)
    data['MonthlyRevenue_label'] = pd.cut(data.MonthlyRevenue, 8, labels=range(8)).astype(int)
    data['CurrentEquipmentDays_label'] = pd.cut(data.CurrentEquipmentDays, 8, labels=range(8)).astype(int)
    data['IncomeGroup_label'] = pd.cut(data.IncomeGroup, 8, labels=range(8)).astype(int)
    data = data.drop(columns=['TotalRecurringCharge', 'MonthsInService', 'AgeHH1', 'AgeHH2', 'MonthlyMinutes', 'MonthlyRevenue', 'CurrentEquipmentDays', 'IncomeGroup'])
    return data

def create_new_feature_with_clustering(data: pd.DataFrame, n_cluster):
    standard_scaler = StandardScaler()
    scaled_data = data.copy()
    scaled_data.loc[:, (scaled_data.dtypes==float)] = standard_scaler.fit_transform(scaled_data.loc[:, (scaled_data.dtypes==float)])
    kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(scaled_data.drop(columns=['Churn']))
    data['cluster_group_label'] = kmeans.labels_
    return data

def preprocessed_data(df: pd.DataFrame, is_fillna=False, drop_columns=['CustomerID', 'ServiceArea'], con_to_cat=False, clustering=False):
    """
    데이터 전처리 함수(원본유지)

    Parameters
    ----------
        is_fillna : 기본값 False
            `False`면 dropna실행 `True`면 Median값으로 채움
        drop_columns : 드랍할 컬럼명 [ ]
        con_to_cat : 연속형 변수를 범주형으로 바꿈
        clustering : clustring을 통한 새로운 Feature생성
    """
    data = df.copy()
    if not is_fillna:
        data = data.dropna()
    else:
        data = fillna_median(data)

    data = data.drop(columns=drop_columns)
    if con_to_cat:
        data = convert_continuous_to_categorical(data)
    data, _ = convert_object_into_integer(data)
    if clustering:
        data = create_new_feature_with_clustering(data, 4)
    
    return data