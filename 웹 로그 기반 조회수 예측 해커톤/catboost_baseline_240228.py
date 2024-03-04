import pandas as pd
import numpy as np
import random
import os
from catboost import CatBoostRegressor, Pool

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


seed = 26
seed_everything(seed)  # Seed 고정

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_train = df_train.drop(['sessionID','userID'],axis=1)
df_test = df_test.drop(['sessionID','userID'],axis=1)

df_train.fillna('NAN', inplace=True)
df_test.fillna('NAN', inplace=True)

df_train['duration_quality'] = df_train['duration']/df_train['quality']
df_test['duration_quality'] = df_test['duration']/df_test['quality']

df_train = df_train.drop(columns = 'bounced')
df_test = df_test.drop(columns = 'bounced')

categorical_features = [
"browser",
"OS",
"device",
"new",
"continent",
"subcontinent",
"traffic_source",
"traffic_medium",
"keyword",
"referral_path",
"country"
]
for i in categorical_features:
    df_train[i] = df_train[i].astype('category')
    df_test[i] = df_test[i].astype('category')

# 시간 단위로 변경
df_train['duration'] = df_train['duration']/3600
df_test['duration'] = df_test['duration']/3600

df_X_train = df_train.drop('TARGET', axis=1)
df_y_train = df_train['TARGET']

# CatBoost 모델 학습
model = CatBoostRegressor(random_state=seed, verbose=False)
full_train_pool = Pool(data = df_X_train, label = df_y_train, cat_features= categorical_features)
model.fit(full_train_pool)

# Test 데이터로 Pool 객체 생성
test_pool = Pool(data=df_test, cat_features=categorical_features)

# Test 데이터에 대한 예측
test_pred = model.predict(test_pool)


# Feature Importance 구하기
feature_importances = model.get_feature_importance()

# Feature Importance를 DataFrame으로 변환
features = pd.DataFrame({
    'Feature': df_X_train.columns,
    'Importance': feature_importances
})

print(features)


test_pred = [0 if i < 0 else i for i in test_pred]

# Sample Submission 업데이트
df_submit = pd.read_csv('sample_submission.csv')
df_submit['TARGET'] = test_pred

df_submit.to_csv("catboost_baseline_240228.csv", index=False)