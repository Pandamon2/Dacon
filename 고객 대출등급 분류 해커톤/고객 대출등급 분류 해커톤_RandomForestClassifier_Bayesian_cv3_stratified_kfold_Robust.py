import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from bayes_opt import BayesianOptimization, UtilityFunction
import tqdm
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight


# 그래프 한글 폰트
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

# 데이터 불러오기
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

# info, 결측치 확인
print(train.info())
print(train.isnull().sum())
print(test.isnull().sum())

print(train['근로기간'].value_counts())

# 근로기간 데이터 변경
train['근로기간'] = train['근로기간'].replace({
    '10+ years': '10+ years',
    '10+years': '10+ years',
    '<1 year': '< 1 year',
    '1 years': '< 1 year',
    '3': '3 years'
})

test['근로기간'] = test['근로기간'].replace({
    '10+ years': '10+ years',
    '10+years': '10+ years',
    '<1 year': '< 1 year',
    '1 years': '< 1 year',
    '3': '3 years'
})


# EDA
# 대출금액 분포
plt.figure(figsize=(12, 6))

sns.histplot(train['대출금액'], bins=30, kde=True, color='skyblue', label='대출금액')
plt.title('대출금액 분포')
plt.xlabel('대출금액')
plt.ylabel('빈도')
plt.legend()
plt.show()

# 연간소득 분포
sns.histplot(train['연간소득'], bins=30, kde=True, color='green', label='연간소득')
plt.title('연간소득 분포')
plt.xlabel('연간소득')
plt.ylabel('빈도')
plt.legend()
plt.show()

sns.histplot(np.log1p(train['연간소득']), bins=30, kde=True, color='green', label='연간소득')
plt.title('연간소득 분포')
plt.xlabel('연간소득')
plt.ylabel('빈도')
plt.legend()
plt.show()

# 총계좌수 분포
plt.figure(figsize=(12, 6))

sns.histplot(train['총계좌수'], bins=30, kde=True, color='blue', label='총계좌수')
plt.title('총계좌수 분포')
plt.xlabel('총계좌수')
plt.ylabel('빈도')
plt.legend()
plt.show()

# 근로기간 Unknown -> 10+ years로 변경(최빈값)
train['근로기간'].replace({'Unknown' : '10+ years'}, inplace = True)
test['근로기간'].replace({'Unknown' : '10+ years'}, inplace = True)

# '근로기간' 열의 값 변환 함수 정의
def convert_experience(experience):
    if pd.isnull(experience) or experience == 'Unknown':
        return -1  # 또는 다른 특별한 값으로 대체할 수 있음
    elif '+' in experience:
        return 10
    elif '< 1' in experience:
        return 0
    else:
        return int(''.join(filter(str.isdigit, experience)))

# '근로기간' 열 변환
train['근로기간'] = train['근로기간'].map(convert_experience)
test['근로기간'] = test['근로기간'].map(convert_experience)

# 변환 후의 값 확인
print(train['근로기간'].value_counts())

# test 대출목적 데이터 변경
test['대출목적'] = test['대출목적'].replace({'결혼' : '기타'})

# 파생변수 생성
# '대출기간'에서 숫자만 추출하여 월 상환액 계산
train['대출기간(month)'] = train['대출기간'].str.extract('(\d+)').astype(int)
test['대출기간(month)'] = test['대출기간'].str.extract('(\d+)').astype(int)

train['월상환액'] = train['대출금액'] / train['대출기간(month)']
test['월상환액'] = test['대출금액'] / test['대출기간(month)']

train['총상환원금 비율'] = train['총상환원금'] / train['대출금액']
test['총상환원금 비율'] = test['총상환원금'] / test['대출금액']
train['총상환이자 비율'] = train['총상환이자'] / train['대출금액']
test['총상환이자 비율'] = test['총상환이자'] / test['대출금액']

train['연간소득_log'] = np.log1p(train['연간소득'])
test['연간소득_log'] = np.log1p(test['연간소득'])

train = train.drop(columns = ['대출기간', '연간소득'], axis= 1)
test = test.drop(columns = ['대출기간', '연간소득'], axis= 1)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler

# 스케일링
scaler = MinMaxScaler()
numeric = train.select_dtypes(exclude= 'object').columns
train[numeric] = scaler.fit_transform(train[numeric])
test[numeric] = scaler.transform(test[numeric])

# 테스트 데이터 ID 추출
test_id = test['ID']

# ID 제거
train = train.drop('ID', axis = 1)
test = test.drop('ID', axis = 1)

# 라벨인코딩
le = LabelEncoder()
object = train.select_dtypes(include = 'object').columns
for i in object:
    if i != '대출등급':
        train[i] = le.fit_transform(train[i])
        test[i] = le.transform(test[i])

# 데이터 분할
X = train.drop('대출등급', axis = 1)
y = train['대출등급']
X_train, X_val, y_train, y_val = train_test_split(X, y , test_size = 0.2, random_state=2024)


# 베이지안 최적화를 위한 목표 함수를 정의합니다.
def rf_cv(n_estimators, min_samples_split, max_features, max_depth):
    # 클래스 가중치 계산
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)

    # 가중치를 딕셔너리로 변환
    class_weight_dict = {class_label: weight for class_label, weight in zip(np.unique(y), class_weights)}

    model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        min_samples_split=int(min_samples_split),
        max_features=min(max_features, 0.999),  # float type
        max_depth=int(max_depth),
        random_state=2024,
        n_jobs=-1,
        class_weight=class_weight_dict  # 클래스 가중치 설정
    )

    # StratifiedKFold를 사용한 3-fold 교차 검증
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=2024)

    # cross_val_score를 사용하여 교차 검증 수행
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro')

    # 교차 검증 점수의 평균을 반환
    return scores.mean()

# 베이지안 최적화 객체를 생성합니다.
bayes_optimizer = BayesianOptimization(
    f=rf_cv,
    pbounds={
        'n_estimators': (100, 500),
        'min_samples_split': (2, 10),
        'max_features': (0.1, 1),
        'max_depth': (10, 50),
    },
    random_state=2024
)

# UtilityFunction 인스턴스를 생성합니다.
utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.01)

# 베이지안 최적화를 수행합니다.
for _ in tqdm.tqdm(range(50)):
    next_point = bayes_optimizer.suggest(utility)
    target = rf_cv(**next_point)
    bayes_optimizer.register(params=next_point, target=target)

# 최적의 하이퍼파라미터를 출력합니다.
print(bayes_optimizer.max)

# 최적의 하이퍼파라미터를 적용하여 모델을 학습합니다.
best_params = bayes_optimizer.max['params']
best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['min_samples_split'] = int(best_params['min_samples_split'])
best_params['max_depth'] = int(best_params['max_depth'])
model = RandomForestClassifier(**best_params, random_state=2024, n_jobs=-1)

# 전체 데이터를 사용하여 모델을 학습합니다.
model.fit(X, y)

# 최적의 하이퍼파라미터를 적용한 모델로 예측을 수행합니다.
# y_pred = model.predict(X_val)
# macro_f1 = f1_score(y_val, y_pred, average='macro')
# print('Macro f1 score (Validation):', macro_f1)

# 테스트 데이터에 대한 예측 생성
y_test_pred = model.predict(test)

# 예측 결과를 데이터프레임으로 만들기
submission = pd.DataFrame({'ID': test_id, '대출등급': y_test_pred})

# 결과 확인
print(submission)

# 예측 결과를 CSV 파일로 저장
submission.to_csv('submission_RandomForestClassifier_bayesian_optimization_cv3_StratifiedKFold_non_split_class_weight.csv', index=False, encoding='utf-8-sig')


# model.fit(X_train, y_train)

# # 최적의 하이퍼파라미터를 적용한 모델로 예측을 수행합니다.
# y_pred = model.predict(X_val)
# macro_f1 = f1_score(y_val, y_pred, average='macro')
# print('Macro f1 score:', macro_f1)
#
# # 테스트 데이터에 대한 예측 생성
# y_test_pred = model.predict(test)
#
# # 예측 결과를 데이터프레임으로 만들기
# submission = pd.DataFrame({'ID': test_id, '대출등급': y_test_pred})
#
# # 결과 확인
# print(submission)
#
# # 예측 결과를 CSV 파일로 저장
# submission.to_csv('submission_RandomForestClassifier_bayesian_optimization_cv3_StratifiedKFold_robust.csv', index=False, encoding='utf-8-sig')
