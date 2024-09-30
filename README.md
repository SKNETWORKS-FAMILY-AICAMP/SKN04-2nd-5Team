# SKN04-2nd-5Team
# 👑Class Dom👑
<p align="center"><img src="./classdom/image/classdom.jpg" width="1000" height="300"/></p>

<hr>

### 🤗 팀명 : 골골대조
 
### 🤭 팀원

<p align="center">
	<img src="./classdom/image/min.jpg" width="200" height="200"/>
	<img src="./classdom/image/seung.jpg" width="200" height="200"/>
	<img src="./classdom/image/su.jpg" width="200" height="200"/>
	<img src="./classdom/image/hye.jpg" width="200" height="200"/>
	<img src="./classdom/image/hye.jpg" width="200" height="200"/>
</p>

<div align="center">
	
|&nbsp;&nbsp; &nbsp; &nbsp; &nbsp;  &nbsp;  &nbsp; 🐶 박진효  &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp;|&nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp; 🐱 고유림  &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp; |&nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp; 🐹 이진섭  &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp; |  &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp; 🐰 이호재  &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp;|&nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp; 🐱 전욱진  &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp; |
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
pip install -r requirements.txt
```

<hr>

### Usage
**이 코드를 실행하기 위해 어떠한 코드를 어떻게 실행해야 하는지 작성**

```cmd
python main.py
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

| ![data.png](attachment:data.png) | ![image-2.png](attachment:image-2.png) |
|----------------------------------|----------------------------------------|

<br>

<hr>

### EDA

#### 1. Churn에 따른 시각화
![타겟에 따른 시각화.png](<attachment:타겟에 따른 시각화.png>)

#### 2. XGBoost Feature Importance에 따른 feature 시각화
- recall이 가장 높은 데이터의 Feature Importance에 따라 상위 3개에 대한 시각화<br>
<br>
![xgboost feature importance 기준 상위 3개 시각화.png](<attachment:xgboost feature importance 기준 상위 3개 시각화.png>)
<br>
<br>
- countplot과 kdeplot을 보았을 때 개별 feature에 대한 Churn의 Yes의 개수가 No보다 많은 것이 없으며, 비율도 동일한 비율만 존재하여 1번의 데이터로는 Churn의 Yes, No를 구분할 방법이 없는 것으로 판단됩니다.

#### 3. 새로운 Feature 생성

**3-1. 연속형 데이터를 범주형 데이터로 생성**<br>
<br>

![스크린샷 2024-09-30 103817.png](<attachment:스크린샷 2024-09-30 103817.png>)

**3-2. Clustering을 통한 새로운 label 생성**<br>
- KMeans 활용<br>
<br>

![image.png](attachment:image.png)

#### 4. Type 변경
- 3번에서 2가지 방법으로 새로운 feature을 생성하였지만 EDA상 유의미한 정보를 찾지 못하여 실행합니다.<br>
<br>

![imshow_type 변경.png](<attachment:imshow_type 변경.png>)

- Heatmap을 통하여 Churn과 다른 feature들간에 유의미한 상관관계는 보이지 않습니다.
<br>
<br>
> Churn과 feature간의 유의미한 관계가 있는 데이터는 없다고 판단됩니다. <br>
> 따라서 기본적인 model을 실행했을 때의 예상되는 Score은 둘의 비율인 0.71에 근접하는 결과가 나올 것 같습니다.<br>
> target의 데이터가 불균일하기 때문에 accuracy보다 precision과 recall의 결과에 대하여 주의해야할 것 같습니다.

<hr>

### Modeling


<hr>

### 수행 결과

#### Machine Learning

**XGBoost**
**1. 모델 평가 결과** <br>

| ![xgboost_평가_fold4.png](attachment:xgboost_평가_fold4.png) | ![xgboost_평가_fold5.png](attachment:xgboost_평가_fold5.png) |
|-----------------------|-----------------------|

**2. Feature Importance** <br>
- recall이 가장 높은 데이터의 feature importance를 확인합니다. <br>
- 위의 과정을 거친 후에 중요도가 낮은 feature부터 하나씩 제거하면서 비교합니다. <br>

| ![xgboost_drop_1.png](attachment:xgboost_drop_1.png) | ![xgboost_drop_2.png](attachment:xgboost_drop_2.png) | ![xgboost_drop_3.png](attachment:xgboost_drop_3.png) |
|-----------------------|-----------------------|-----------------------|

**LightGBM**
**1. 모델 평가 결과** <br>

| ![lgbm_result_1.png](attachment:lgbm_result_1.png) | ![lgbm_result_2.png](attachment:lgbm_result_2.png) | ![lgbm_result_3.png](attachment:lgbm_result_3.png) |
|-----------------------|-----------------------|-----------------------|

**2. Feature Importance** <br>
- recall이 가장 높은 데이터의 feature importance를 확인합니다. <br>
- 위의 과정을 거친 후에 중요도가 낮은 feature부터 하나씩 제거하면서 비교합니다. <br>

| ![lgbm_drop_1.png](attachment:lgbm_drop_1.png) | ![lgbm_drop_10.png](attachment:lgbm_drop_10.png) | ![lgbm_drop_20.png](attachment:lgbm_drop_20.png) |
|-----------------------|-----------------------|-----------------------|

-lgbm 기본 vs importance 낮은 것들 drop하면서 실행했을 때 -> recall이 변화가 없음 유의미한
- 

> XGBoost와 LGBM의 각각 기존 

#### Deep Learning



<hr>

### 한 줄 회고

***회고 작성***