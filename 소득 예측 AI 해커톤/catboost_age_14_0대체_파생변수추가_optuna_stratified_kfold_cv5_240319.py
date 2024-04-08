import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_squared_error
import optuna

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

# 결측치 확인
print(train.isnull().sum())
print(test.isnull().sum())

# 해당 컬럼 확인
print(train['Household_Status'].value_counts())

# 결측치 대체(최빈값)
test['Household_Status'] = test['Household_Status'].fillna('Householder')

train_age_14 = train[train['Age'] <= 14]
print(train_age_14['Income'].value_counts())

# ID 추출
X_train = train.drop(columns=['ID', 'Income'])
y_train = train['Income']
X_test = test.drop(columns=['ID'])


# 파생변수 추가
# Occupation 파생변수
def get_occupation_group(occupation):
    # 직업 카테고리에 따라 그룹을 할당하는 함수
    if occupation in ['Professional', 'Management']:
        return 'Managerial & Professional'
    elif occupation in ['Admin Support (include Clerical)', 'Management']:
        return 'Admin & Management'
    elif occupation in ['Services', 'Private Household Services']:
        return 'Services'
    elif occupation == 'Sales':
        return 'Sales & Marketing'
    elif occupation in ['Technical', 'Machine Operators & Inspectors', 'Technicians & Support']:
        return 'Technical & Support'
    elif occupation == 'Craft & Repair':
        return 'Craft & Repair'
    else:
        return 'Other'

# 직업 그룹 파생 변수 생성
X_train['Occupation_Group'] = X_train['Occupation_Status'].apply(get_occupation_group)
X_test['Occupation_Group'] = X_test['Occupation_Status'].apply(get_occupation_group)


# Industry 파생변수
# 산업 부문에 따라 그룹 할당
def get_industry_group(industry):
    # 산업 부문에 따라 그룹을 할당하는 함수
    if 'Manufacturing' in industry:
        return 'Manufacturing'
    elif 'Retail' in industry:
        return 'Retail'
    elif 'Services' in industry:
        return 'Services'
    elif 'Transportation' in industry:
        return 'Transportation'
    else:
        return 'Other'

# 산업 부문 그룹 파생 변수 생성
X_train['Industry_Group'] = X_train['Industry_Status'].apply(get_industry_group)
X_test['Industry_Group'] = X_test['Industry_Status'].apply(get_industry_group)


# Education 파생변수
# 'Education_Status' 컬럼을 기반으로 교육 수준을 그룹화하여 새로운 변수 'Education_Level_Group'을 만듭니다.
def categorize_education_level(education_status):
    if 'High' in education_status:
        return 'High School Graduate'
    elif 'Bachelors degree' in education_status or 'Masters degree' in education_status or 'Doctorate degree' in education_status:
        return 'Higher Education'
    elif 'Associates degree' in education_status or 'Professional degree' in education_status:
        return 'Associate/Professional Degree'
    elif 'Middle' in education_status or 'Elementary' in education_status or 'Kindergarten' in education_status:
        return 'Primary/Middle School'
    else:
        return 'Other'

X_train['Education_Level_Group'] = X_train['Education_Status'].apply(categorize_education_level)
X_test['Education_Level_Group'] = X_test['Education_Status'].apply(categorize_education_level)


# 'Occupation_Status'와 'Industry_Status'를 결합하여 새로운 변수 'Occupation_Industry_Group'을 만듭니다.
X_train['Occupation_Industry_Group'] = X_train['Occupation_Status'] + ' - ' + X_train['Industry_Status']
X_test['Occupation_Industry_Group'] = X_test['Occupation_Status'] + ' - ' + X_test['Industry_Status']



# 수치형 변수
num_columns = X_train.select_dtypes(exclude='object').columns

# 연속형 변수들에 로그 변환 적용
for column in num_columns:
    X_train[column] = np.log1p(X_train[column])
    X_test[column] = np.log1p(X_test[column])

# 스케일링
scaler = MinMaxScaler()
X_train[num_columns] = scaler.fit_transform(X_train[num_columns])
X_test[num_columns] = scaler.transform(X_test[num_columns])

# 모든 object 열을 범주형으로 변환
for column in X_train.select_dtypes(include='object').columns:
    X_train[column] = X_train[column].astype('category')
    X_test[column] = X_test[column].astype('category')

# 범주형 변수의 열 이름을 리스트로 변환
categorical_columns = X_train.select_dtypes(include='category').columns.tolist()


# Objective 함수 정의
def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'random_seed': trial.suggest_int('random_seed', 0, 1000),
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'verbose': False
    }

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_index, val_index in kf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

        train_pool = Pool(X_tr, label=y_tr, cat_features=categorical_columns)
        val_pool = Pool(X_val, label=y_val, cat_features=categorical_columns)

        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=False)

        val_pred = model.predict(X_val)
        rmse = mean_squared_error(y_val, val_pred, squared=False)
        scores.append(rmse)

    return np.mean(scores)


# Optuna 스터디 실행
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# 최적 하이퍼파라미터 추출
best_params = study.best_params

# 최적 하이퍼파라미터를 사용하여 CatBoost 모델 학습 및 평가
model = CatBoostRegressor(**best_params)
model.fit(X_train, y_train, cat_features=categorical_columns)


# 테스트 데이터에 대한 예측 수행
test_predictions = model.predict(X_test)
test_predictions = np.maximum(test_predictions, 0)

# Age가 14 이하인 경우 해당 인덱스를 기억
indices_to_zero = test[test['Age'] <= 14].index

# Age가 14 이하인 경우 예측값을 0으로 설정
test_predictions[indices_to_zero] = 0

submission = pd.read_csv('./sample_submission.csv')
submission['Income'] = test_predictions

submission.to_csv('catboost_age_14_0대체_파생변수추가_optuna_stratified_kfold_cv5_240319.csv', index=False)

print(submission)