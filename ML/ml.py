
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from data.preprocessed_data import preprocessed_data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report,  confusion_matrix
from bayes_opt import BayesianOptimization
import xgboost
import matplotlib.pyplot as plt


data = pd.read_csv('../data/train.csv')
drop_na_data = preprocessed_data(data)

def xgboost_result(drop_na_data):
    best_param = {'learning_rate': 0.1005721895883048, 'max_depth': 5, 'max_leaves': 494, 'n_estimators': 194}
    drop_na_train, temp = train_test_split(drop_na_data, test_size=0.4, random_state=0)
    drop_na_valid, drop_na_test = train_test_split(temp, test_size=0.5, random_state=0)
    xgb = xgboost.XGBClassifier(
                **best_param,
                random_state=0
            )
    xgb.fit(
        drop_na_train.drop(columns=['Churn']), drop_na_train['Churn'],
        eval_set=[(drop_na_valid.drop(columns=['Churn']), drop_na_valid['Churn'])],
        verbose=0
    )
    plt.rcParams["figure.figsize"] = (10, 15)
    xgboost.plot_importance(xgb)
    importance_data = pd.DataFrame({'importance':xgb.feature_importances_}, index = xgb.get_booster().feature_names)
    importance_data = importance_data.sort_values('importance')
    drop_na_train, temp = train_test_split(drop_na_data, test_size=0.4, random_state=0)
    drop_na_valid, drop_na_test = train_test_split(temp, test_size=0.5, random_state=0)
    max_drop_count = 20
    results = {}
    for i in range(0, max_drop_count):
        col = importance_data.iloc[:i, :].index

        xgb = xgboost.XGBClassifier(
                    **best_param,
                    random_state=0
                )
        xgb.fit(
            drop_na_train.drop(columns=['Churn']).drop(columns=col), drop_na_train['Churn'],
            eval_set=[(drop_na_valid.drop(columns=['Churn']).drop(columns=col), drop_na_valid['Churn'])],
            verbose=0
        )
        xgb_predict = xgb.predict(drop_na_test.drop(columns=['Churn']).drop(columns=col))
        results.update({
            f'drop {i+1} column':{
                classification_report(drop_na_test['Churn'], xgb_predict)
            }
        })
    for i, result in results.items():
        print(i)
        for r in result:
            print(r)

