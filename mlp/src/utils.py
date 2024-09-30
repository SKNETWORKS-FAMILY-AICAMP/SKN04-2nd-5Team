import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

def convert_category_into_integer(df: pd.DataFrame, columns: list):
    """
    주어진 DataFrame의 특정 열들을 범주형에서 정수형으로 변환합니다.
    
    Parameters:
    - df (pd.DataFrame): 변환할 데이터프레임
    - columns (list): 범주형에서 정수형으로 변환할 열 이름의 리스트
    
    Returns:
    - pd.DataFrame: 변환된 데이터프레임
    - dict: 각 열에 대해 적합한 LabelEncoder 객체를 포함하는 딕셔너리
    """
    label_encoders = {}  # 각 열의 LabelEncoder 객체를 저장할 딕셔너리입니다.
    
    for column in columns:
        # 각 열에 대해 LabelEncoder 객체를 생성합니다.
        label_encoder = LabelEncoder()
        
        # LabelEncoder를 사용하여 해당 열의 범주형 데이터를 정수형으로 변환합니다.
        df.loc[:, column] = label_encoder.fit_transform(df[column])
        
        # 변환된 LabelEncoder 객체를 딕셔너리에 저장합니다.
        label_encoders.update({column: label_encoder})
    
    # 변환된 데이터프레임과 LabelEncoder 객체를 포함하는 딕셔너리를 반환합니다.
    return df, label_encoders

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Classification report 출력
    print(classification_report(y_true, y_pred))