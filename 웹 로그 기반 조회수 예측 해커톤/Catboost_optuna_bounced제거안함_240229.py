import pandas as pd
import numpy as np
import random
import os
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
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

# df_train = df_train.drop(columns = 'bounced')
# df_test = df_test.drop(columns = 'bounced')

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
    "country",
    "bounced"
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

    train_pool = Pool(data=X_train, label=y_train, cat_features=categorical_features)
    val_pool = Pool(data=X_val, label=y_val, cat_features=categorical_features)

    catboost_model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose_eval=100)

    val_pred = catboost_model.predict(val_pool)

    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    return rmse

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

df_submit.to_csv("CatBoost_optuna_bounced제거안함_240229.csv", index=False)