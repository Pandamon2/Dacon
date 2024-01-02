import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import numpy as np
from hyperopt import fmin, tpe, hp
from sklearn.metrics import mean_absolute_error

# 데이터 불러오기
train_df = pd.read_csv("C:/Users/lime/OneDrive/바탕 화면/open/train.csv")
submission_df = pd.read_csv("C:/Users/lime/OneDrive/바탕 화면/open/sample_submission.csv")
test = pd.read_csv("C:/Users/lime/OneDrive/바탕 화면/open/23년 평균기온 데이터.csv", encoding = 'euc-kr')

# 일조합 null값 고려
train_df = train_df[train_df['일시'] >= '1973-04-16']

# 날짜 데이터 변환
train_df['일시'] = pd.to_datetime(train_df['일시'])
train_df = train_df.set_index('일시')

# 데이터의 시간 간격 지정
train_df.index.freq = 'D'

# prophet에서 데이터를 인식하도록 일시는 ds로, target값인 평균기온은 y로 지정해줍니다.
train_df = train_df.reset_index()
train_df = train_df.rename(columns={'일시': 'ds', '평균기온': 'y'})

# 목적 함수 정의 (교차 검증을 통한 MAE 계산)
def objective(params):
    prophet = Prophet(changepoint_prior_scale=params['changepoint_prior_scale'],
                      seasonality_prior_scale=params['seasonality_prior_scale'],
                      seasonality_mode=params['seasonality_mode'])
    prophet.fit(train_df)

    cv_results = cross_validation(prophet, initial='730 days', period='180 days', horizon='358 days')
    mae = performance_metrics(cv_results)['mae'].values[0]

    print(f"Current Hyperparameters: {params}, Current MAE: {mae}")

    return mae

# 하이퍼파라미터 탐색 공간 정의
space = {
    'changepoint_prior_scale': hp.uniform('changepoint_prior_scale', 0.001, 0.03),
    'seasonality_prior_scale': hp.uniform('seasonality_prior_scale', 0.1, 3.0),
    'seasonality_mode': hp.choice('seasonality_mode', ['additive', 'multiplicative'])
}

# 베이지안 최적화 수행
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50)

# 최적의 하이퍼파라미터로 최종 모델 훈련
prophet = Prophet(changepoint_prior_scale=best['changepoint_prior_scale'],
                  seasonality_prior_scale=best['seasonality_prior_scale'],
                  seasonality_mode=['additive', 'multiplicative'][best['seasonality_mode']])
prophet.fit(train_df)

# 모델 예측
future_data = prophet.make_future_dataframe(periods=358, freq='d')  # periods는 예측할 기간
forecast_data = prophet.predict(future_data)

submission_df['평균기온'] = forecast_data.yhat[-358:].values

submission_df_4 = submission_df[submission_df['일시'] < '2023-05-01']
test_4 = test[test['일시'] < '2023-05-01']

mae_4 = mean_absolute_error(submission_df_4['평균기온'], test_4['평균기온'])
mae_12 = mean_absolute_error(submission_df['평균기온'], test['평균기온'])
print(mae_4)
print(mae_12)


# 결과 저장
submission_df.to_csv("prophet_bayesian_optimization_231226.csv", index=False, encoding='utf-8-sig')