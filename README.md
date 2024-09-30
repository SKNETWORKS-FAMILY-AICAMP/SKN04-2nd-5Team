# SKN04-2nd-5Team
# 👑Class Dom👑
<p align="center"><img src="./classdom/image/classdom.jpg" width="1000" height="300"/></p>

<hr>

### 🤗 팀명 : 골골대조
 
### 🤭 팀원


<div align="center">
	
|&nbsp;&nbsp; &nbsp; &nbsp; &nbsp;  &nbsp;  &nbsp; 🐶 박진효  &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp;|&nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp; 🐱 고유림  &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp; |&nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp; 🐹 이진섭  &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp; |  &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp; 🐰 이호재  &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp;|&nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp; 🐱 전욱진  &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp; |
| DL | ML | DL | ML | EDA |
|------------------------------------------|--------------------------------------|------------------------------------------|-----------------------------------|--------------------------------------|
 
</div>

<hr>

### 👨‍🏫 프로젝트 개요
이 프로젝트는 머신러닝, 딥러닝을 활용하여 기업의 고객 이탈을 예측하는 모델을 구축합니다. 이를 통하여 기업은 고객 충성도를 높이고, 맞춤형 마케팅 전략을 수립할 수 있습니다.

<hr>

### 👩‍🏫 프로젝트 목표
이 프로젝트의 주요 목표는 고객 이탈 예측 모델의 정확성을 극대화하여 기업의 의사결정을 지원하는 것입니다. 이를 위해 다양한 데이터 분석 기법과 머신러닝 알고리즘을 비교적용하여 최적의 모델을 개발합니다. 또한, 예측 결과를 바탕으로 고객 유지 전략을 제안하여 실질적인 비즈니스 성과를 향상시키고자 합니다. 궁극적으로는 고객 경험을 개선하고, 장기적인 고객 관계를 구축하는 것이 목표입니다.

<hr>

### 🔨 기술 스택
<div>
<img src="https://img.shields.io/badge/git-F05032?style=for-the-badge&logo=git&logoColor=white">
<img src="https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white">
<img src="https://camo.githubusercontent.com/0d0779a129f1dcf6c31613b701fe0646fd4e4d2ed2a7cbd61b27fd5514baa938/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f707974686f6e2d3336373041303f7374796c653d666f722d7468652d6261646765266c6f676f3d707974686f6e266c6f676f436f6c6f723d666664643534">
</div>

<hr>

### Prerequisites
**이 프로젝트를 실행하기 위해 필요한 패키지 등을 정의**

```cmd
pip install numpy
pip install pandas
pip install 
```

<hr>

### Usage

- Machine Learning
```cmd
python ./ML/ml.py
```
- Deep Learning
```cmd
python train.py
```

<hr> 

### Data

**1. NaN값 제거합니다.** <br>
<br>

**2. NaN을 해당 컬럼의 median값으로 채웁니다.** <br>
<br>

**3. 2번 데이터에 일부 연속형 데이터를 범주형 데이터로 변환합니다.** <br>
<br>	

**4. 2번 데이터에 클러스터링을 통한 라벨 feature를 추가합니다.**
<br>

|![data](https://github.com/user-attachments/assets/f4924aeb-f8cd-4873-804e-35abd535c010)|![data2](https://github.com/user-attachments/assets/997d3b34-db4d-4b90-824f-ac31b70cedb7)|
|----------------------------------|----------------------------------------|

<br>

<hr>

### EDA

#### 1. Churn에 따른 시각화
![타겟에 따른 시각화](https://github.com/user-attachments/assets/2e3d1aa8-d56b-44bb-a922-83e3b28da42a)

#### 2. XGBoost Feature Importance에 따른 feature 시각화
- recall이 가장 높은 데이터의 Feature Importance에 따라 상위 3개에 대한 시각화<br>
<br>

![xgboost feature importance 기준 상위 3개 시각화](https://github.com/user-attachments/assets/627f2f01-a3e0-4b1e-ad50-a56f4682dccf)
<br>
<br>
- countplot과 kdeplot을 보았을 때 개별 feature에 대한 Churn의 Yes의 개수가 No보다 많은 것이 없으며, 비율도 동일한 비율만 존재하여 1번의 데이터로는 Churn의 Yes, No를 구분할 방법이 없는 것으로 판단됩니다.

#### 3. 새로운 Feature 생성

**3-1. 연속형 데이터를 범주형 데이터로 생성**<br>
<br>

![스크린샷 2024-09-30 103817](https://github.com/user-attachments/assets/4533206a-b206-43e1-8a62-50566bd6111f)

**3-2. Clustering을 통한 새로운 label 생성**<br>
- KMeans 활용<br>
<br>

![연속에서 범주형 데이터로 변환](https://github.com/user-attachments/assets/511b0819-a5a5-434b-962f-1b58a83c1f80)

#### 4. Type 변경
- 3번에서 2가지 방법으로 새로운 feature을 생성하였지만 EDA상 유의미한 정보를 찾지 못하여 실행합니다.<br>
<br>

![imshow_type 변경](https://github.com/user-attachments/assets/c4874927-58b2-4fa3-bf98-93b9faa04ae4)

- Heatmap을 통하여 Churn과 다른 feature들간에 유의미한 상관관계는 보이지 않습니다.
<br>

#### 5. 소거법

**5-1. 기존 Feature 제거를 통한 데이터 혼잡성 개선 시도** <br>
<br>
유사성 있는 데이터 소거: Service Area/Prizm Code, Occupation/CreditRating/Incomegroup 등 유사성있는 데이터,<br>
혹은 Churn User Demographic 분석에 중요도 떨어지는 데이터 제거하였습니다.<br>
"TotalRecurringCharge, MonthsInService, AgeHH1, AgeHH2, MonthlyMinutes, MonthlyRevenue, CurrentEquipmentDays, Incomegroup"<br>
Upper, lowerquartile 제거하여 데이터 개선 여지 체크합니다.<br>

<br>
- Churn과 feature간의 유의미한 관계가 있는 데이터는 없다고 판단됩니다. <br>
따라서 기본적인 model을 실행했을 때의 예상되는 Score은 둘의 비율인 0.71에 근접하는 결과가 나올 것 같습니다.<br>
target의 데이터가 불균일하기 때문에 accuracy보다 precision과 recall의 결과에 대하여 주의해야할 것 같습니다.

<hr>

### Modeling

#### Machine Learning

- 데이터 전처리 후 분류 task에 적합하고 accuracy를 잘 나오는 모델을 선택하였습니다.

#### Deep Learning

- 모델 선정 기준: 시계열 관련 데이터가 없어 RNN, LSTM 제외, 그 외에도 데이터가 복잡하지 않아(이미지, 영상 등이 아니므로) MLP 선정했습니다.

- 문제 및 해결방안:
    - 정확도는 잘 나왔으나 loss수치가 0.58로 높았습니다. <br>
    1. batch_size, learning_rate, hidden_dim, dropout_ratio 등 수동으로 parameter를 조정했습니다. <br>
    1-1. layer수를 2개에서 3개로 늘렸습니다. <br>
    1-2. Optimizer Adam에서 AdamW로 변경하였으나 유의미한 차이가 없었습니다.<br>
    <br>
    2. recall이 1.0이라는 수치가 나오는 현상이 발생했습니다.<br>
    2-1. output_dim을 1에서 2로 늘리고 softmax()를 사용하였으며, 손실함수를ㄹ BCE에서 CE로 변경했습니다.<br>
    <br>
    3. nni를 통한 parameter tuning에서 문제가 있었습니다.

<hr>

### 수행 결과

#### Machine Learning

**XGBoost**
**1. 모델 평가 결과** <br>

![xgboost_result1](https://github.com/user-attachments/assets/b6b2a3d4-b631-4e9b-b6cb-11237f5139ae)

**2. Feature Importance** <br>
- recall이 가장 높은 데이터의 feature importance를 확인합니다. <br>
- 위의 과정을 거친 후에 중요도가 낮은 feature부터 하나씩 제거하면서 비교합니다. <br>

|![xgboost_feature_drop1](https://github.com/user-attachments/assets/f72f8914-defa-4a03-b831-5395fe253233)|![xgboost_feature_drop5](https://github.com/user-attachments/assets/c305ee92-23ae-4764-8d06-872a9c05bf16)|![xgboost_feature_drop20](https://github.com/user-attachments/assets/709448ff-5dc8-4194-adbf-bd91315e1414)|
|-----------------------|-----------------------|-----------------------|

<br>
<br>

**LightGBM**
<br>
**1. 모델 평가 결과** <br>

![lightgbm_result](https://github.com/user-attachments/assets/dccef868-f87f-41bc-acb1-a9b684a1281b)

**2. Feature Importance** <br>
- recall이 가장 높은 데이터의 feature importance를 확인합니다. <br>
- 위의 과정을 거친 후에 중요도가 낮은 feature부터 하나씩 제거하면서 비교합니다. <br>

|![lgbm_drop1](https://github.com/user-attachments/assets/5b32d1c6-494d-44b4-a198-9fc8e19d5cac)|![lgbm_drop5](https://github.com/user-attachments/assets/bb091f7e-ea7c-4d65-acca-6e7947e8f5de)|![lgbm_drop20](https://github.com/user-attachments/assets/c7ef9523-10d1-4ab7-828e-3d0e5dca982c)|
|-----------------------|-----------------------|-----------------------|

> XGBoost와 LGBM의 각각 기존 데이터로 모델을 실행하였을 때 recall값에서 유의미한 변화가 없다는 것을 확인할 수 있습니다.

<br>

#### Deep Learning

**Test Dataloader**
![DL_result](https://github.com/user-attachments/assets/d10430e6-ec61-4eec-a59d-acbe61141bf9)
<br>

**nni 결과**

![nni_result](https://github.com/user-attachments/assets/4202742b-1504-4e49-baa3-03993b899b9e)

> 머신러닝에서 XGBoost를 사용하여 정확도가 72% 정도가 나왔으며, <br>
> 딥러닝에서 MLP를 사용하여 정확도가 71% 정도가 나왔습니다. <br>
> 현재/이탈 고객들을 대상으로 만족도 조사를 진행하면 훨씬 좋은 성능을 보일 것으로 기대됩니다.


<hr>

### 한 줄 회고
```
박진효 - 고생 많이 한 팀원들 감사합니다. 부족한 팀장이라 미안합니다.
```
```
고유림 - 어려웠습니다. 한발짝 좀 더 나가아갔던 시간이었습니다.
```
```
이진섭 - 환경 설정에서 여러 이슈를 겪으며 해결하는 데 많은 애를 먹었지만, 앞으로는 더 신중하게 설정해야겠다.
```
```
이호재 - 다들 고생하셨고 주말이껴서 더 힘든 느낌이 드네요.
```
```
전욱진 - 데이터를 아무리 봐도 모르겠어요
```