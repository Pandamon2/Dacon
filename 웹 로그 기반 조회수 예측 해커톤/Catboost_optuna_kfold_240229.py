import pandas as pd
import numpy as np
import random
import os
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
import optuna

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

# 데이터 분할 (Train 데이터를 다시 Train과 Validation으로 나누기)
X_train, X_val, y_train, y_val = train_test_split(df_X_train, df_y_train, test_size=0.2, random_state=seed)

# Optuna를 활용한 CatBoost 모델의 하이퍼파라미터 최적화
def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 500, 2000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-9, 10, log=True),
        'border_count': trial.suggest_int('border_count', 1, 255),
        'thread_count': os.cpu_count(),
        'random_state': seed,
        'verbose': False
    }

    catboost_model = CatBoostRegressor(**params)

    # K-fold 교차 검증
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    rmse_scores = []

    for train_index, val_index in kf.split(X_train):
        X_train_kf, X_val_kf = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_kf, y_val_kf = y_train.iloc[train_index], y_train.iloc[val_index]

        train_pool = Pool(data=X_train_kf, label=y_train_kf, cat_features=categorical_features)
        val_pool = Pool(data=X_val_kf, label=y_val_kf, cat_features=categorical_features)

        catboost_model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose_eval=100)

        val_pred = catboost_model.predict(val_pool)

        rmse = np.sqrt(mean_squared_error(y_val_kf, val_pred))
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# 최적의 하이퍼파라미터로 CatBoost 모델 재학습
best_params = study.best_params
catboost_model = CatBoostRegressor(**best_params)
train_pool = Pool(data=df_X_train, label=df_y_train, cat_features=categorical_features)
catboost_model.fit(train_pool)

# Test 데이터에 대한 예측
final_pred = catboost_model.predict(df_test)

final_pred = [0 if i < 0 else i for i in final_pred]

# Sample Submission 업데이트
df_submit = pd.read_csv('sample_submission.csv')
df_submit['TARGET'] = final_pred

df_submit.to_csv("CatBoost_optuna_kfold_240229.csv", index=False)