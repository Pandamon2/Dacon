import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler


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

# 데이터 변경
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

# Optuna의 목적 함수를 정의합니다.
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'max_features': trial.suggest_float('max_features', 0.1, 1),
        'max_depth': trial.suggest_int('max_depth', 10, 50),
    }

    # class_weight 파라미터를 Optuna로 최적화
    class_weights = trial.suggest_categorical('class_weights', [None, 'balanced'])

    model = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        min_samples_split=params['min_samples_split'],
        max_features=min(params['max_features'], 0.999),  # float type
        max_depth=params['max_depth'],
        random_state=2024,
        n_jobs=-1,
        class_weight=class_weights
    )

    # Define the StratifiedKFold
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True)

    # Use StratifiedKFold in cross_val_score
    score = cross_val_score(model, X, y, cv=stratified_kfold, scoring='f1_macro').mean()

    return score

# Optuna 스터디를 생성합니다.
study = optuna.create_study(direction='maximize', study_name='RandomForestOptimization')

# 목적 함수를 최적화합니다.
study.optimize(objective, n_trials=50, n_jobs=-1)

# 최적의 하이퍼파라미터를 가져옵니다.
best_params = study.best_params

# 최적의 하이퍼파라미터로 모델을 생성합니다.
best_model = RandomForestClassifier(
    n_estimators=best_params['n_estimators'],
    min_samples_split=best_params['min_samples_split'],
    max_features=min(best_params['max_features'], 0.999),  # float type
    max_depth=best_params['max_depth'],
    random_state=2024,
    n_jobs=-1,
    class_weight=best_params['class_weights']
)

# 전체 학습 데이터로 모델을 학습합니다.
best_model.fit(X, y)

# 최종 테스트 세트에 대한 예측을 수행합니다.
y_test_pred = best_model.predict(test)

# 제출용 데이터프레임을 생성합니다.
submission = pd.DataFrame({'ID': test_id, '대출등급': y_test_pred})

# CSV
submission.to_csv('submission_RandomForestClassifier_optuna_cv5_StratifiedKFold_non_split_class_weight.csv', index=False, encoding='utf-8-sig')