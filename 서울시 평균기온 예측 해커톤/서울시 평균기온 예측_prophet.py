import pandas as pd
from prophet import Prophet
import itertools
import numpy as np
from sklearn.metrics import mean_absolute_error

train_df = pd.read_csv("C:/Users/lime/OneDrive/바탕 화면/open/train.csv")
submission_df = pd.read_csv("C:/Users/lime/OneDrive/바탕 화면/open/sample_submission.csv")

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

# 최적의 하이퍼파라미터로 최종 모델 훈련
prophet = Prophet()

prophet.fit(train_df)

# 모델 예측
future_data = prophet.make_future_dataframe(periods = 358, freq = 'd') #periods는 예측할 기간
forecast_data = prophet.predict(future_data)

submission_df['평균기온'] = forecast_data.yhat[-358:].values
submission_df

submission_df_4 = submission_df[submission_df['일시'] < '2023-05-01']
