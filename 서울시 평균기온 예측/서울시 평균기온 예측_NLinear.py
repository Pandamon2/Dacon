import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from darts.timeseries import TimeSeries
from darts.models import NLinearModel

train_df = pd.read_csv("C:/Users/lime/OneDrive/바탕 화면/open/train.csv")
submission_df = pd.read_csv("C:/Users/lime/OneDrive/바탕 화면/open/sample_submission.csv")
test = pd.read_csv("C:/Users/lime/OneDrive/바탕 화면/open/23년 평균기온 데이터.csv", encoding = 'euc-kr')

df = pd.DataFrame(train_df)
df['일시'] = pd.to_datetime(df['일시'])

# 결측치 처리
df = df.fillna(df.mean())

# TimeSeries 객체 생성
series = TimeSeries.from_dataframe(df, '일시', ['평균기온'])

# NLinearModel
NLinear = NLinearModel(input_chunk_length=709,
                     output_chunk_length=460,
                     n_epochs=1,
                     random_state = 42,
                     batch_size = 293)
NLinear.fit(series)

future = pd.DataFrame()
future['일시'] = pd.date_range(start='2023-01-01', periods=358, freq='D')

# 모델로 예측 수행
forecast = NLinear.predict(len(future['일시']))
pred_y = forecast['평균기온']

# Assuming pred_y is a TimeSeries object
pred_y_values = pred_y.pd_dataframe().values

# Now you can use pred_y_values as a numpy array or convert it to a DataFrame
df_pred_y = pd.DataFrame(pred_y_values, index=pred_y.time_index, columns=pred_y.columns)

submission_df['평균기온'] = df_pred_y['평균기온'].values

submission_df_4 = submission_df[submission_df['일시'] < '2023-05-01']
test_4 = test[test['일시'] < '2023-05-01']

mae_4 = mean_absolute_error(submission_df_4['평균기온'], test_4['평균기온'])
mae_12 = mean_absolute_error(submission_df['평균기온'], test['평균기온'])
print(mae_4)
print(mae_12)