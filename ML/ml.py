
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from data.preprocessed_data import preprocessed_data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report,  confusion_matrix
from bayes_opt import BayesianOptimization
import xgboost
import lightgbm
import matplotlib.pyplot as plt




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
    print(classification_report(drop_na_test['Churn'], xgb.predict(drop_na_test.drop(columns=['Churn']))))
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

def lgbm_result(drop_na_data):
    best_param = {'colsample_bytree': 0.26970343976123257, 'learning_rate': 0.01425085636848397, 'max_depth': 17, 'n_estimators': 789, 'subsample': 0.8830109334221372}
    drop_na_train, temp = train_test_split(drop_na_data, test_size=0.4, random_state=0)
    drop_na_valid, drop_na_test = train_test_split(temp, test_size=0.5, random_state=0)
    lgbm = lightgbm.LGBMClassifier(
                **best_param,
                random_state=0,
                verbose=0
            )
    lgbm.fit(
        drop_na_train.drop(columns=['Churn']), drop_na_train['Churn'],
        eval_set=[(drop_na_valid.drop(columns=['Churn']), drop_na_valid['Churn'])]
    )

    print(classification_report(drop_na_test['Churn'], lgbm.predict(drop_na_test.drop(columns=['Churn']))))
    plt.rcParams["figure.figsize"] = (10, 15)
    lightgbm.plot_importance(lgbm)
    importance_data = pd.DataFrame({'importance':lgbm.feature_importances_}, index = lgbm.feature_names_in_)
    importance_data = importance_data.sort_values('importance')
    drop_na_train, temp = train_test_split(drop_na_data, test_size=0.4, random_state=0)
    drop_na_valid, drop_na_test = train_test_split(temp, test_size=0.5, random_state=0)
    max_drop_count = 20
    results = {}
    for i in range(0, max_drop_count):
        col = importance_data.iloc[:i, :].index

        lgbm = lightgbm.LGBMClassifier(
                    **best_param,
                    random_state=0,
                    verbose=0
                )
        lgbm.fit(
            drop_na_train.drop(columns=['Churn']).drop(columns=col), drop_na_train['Churn'],
            eval_set=[(drop_na_valid.drop(columns=['Churn']).drop(columns=col), drop_na_valid['Churn'])]
        )
        lgbm_predict = lgbm.predict(drop_na_test.drop(columns=['Churn']).drop(columns=col))
        results.update({
            f'drop {i+1} column':{
                classification_report(drop_na_test['Churn'], lgbm_predict)
            }
        })
    for i, result in results.items():
        print(i)
        for r in result:
            print(r)


if __name__ == '__main__':
    data = pd.read_csv('../data/train.csv')
    drop_na_data = preprocessed_data(data)
    print('XGBoost Result=================================================')
    print('='*50)
    xgboost_result(drop_na_data)
    print('\n\n\n')
    print('LightGBM Result=================================================')
    print('='*50)
    lgbm_result(drop_na_data)