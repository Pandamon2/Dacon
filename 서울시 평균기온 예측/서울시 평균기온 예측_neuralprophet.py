import pandas as pd
from neuralprophet import NeuralProphet

# 파일 불러오기
train_df = pd.read_csv("C:/Users/lime/OneDrive/바탕 화면/open/train.csv")
submission_df = pd.read_csv("C:/Users/lime/OneDrive/바탕 화면/open/sample_submission.csv")


# 날짜 데이터 변환
train_df['일시'] = pd.to_datetime(train_df['일시'])
train_df = train_df.set_index('일시')

# 데이터의 시간 간격 지정
train_df.index.freq = 'D'

# 일시 컬럼이 인덱스로 할당됩니다.
train_df.head()

# prophet에서 데이터를 인식하도록 일시는 ds로, target값인 평균기온은 y로 지정해줍니다.
train_df = train_df.reset_index()
train_df = train_df.rename(columns={'일시': 'ds', '평균기온': 'y'})

train_df = train_df[['ds','y']]

train_df

#모델 학습
prophet = NeuralProphet(  # 연간 계절성 사용 여부
    loss_func='mae' # MAE 손실 함수 설정
)
prophet.fit(train_df)

#모델 예측
future_data = prophet.make_future_dataframe(df = train_df,periods = 358) #periods는 예측할 기간
forecast_data = prophet.predict(future_data)

submission_df['평균기온'] = forecast_data['yhat1'].values
submission_df

#결과 저장
submission_df.to_csv("neural_prophet.csv", index=False)