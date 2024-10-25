import pandas as pd
stocks_data['Date'] = pd.to_datetime(stocks_data['Date'])
stocks_data.index = stocks_data['Date']
# nan_data = stocks_data[stocks_data.isnull().any(axis=1)]

for i in ['Adj Close','Close',	'High',	'Low',	'Open',	'Volume']:
    stocks_data[i] = stocks_data[i].interpolate(method='linear')
stocks_data

merged_df = pd.merge(stocks_data, companies_data[['Symbol', 'Sector']], on='Symbol')
merged_df['Date'] = pd.to_datetime(merged_df['Date'])
merged_df.set_index('Date', inplace=True)
merged_data = merged_df.groupby(['Date', 'Sector'])['Adj Close'].mean().unstack()
merged_data

!pip install prophet

import pandas as pd
import matplotlib.pyplot as plt

#2013~2023 -> 2024 (산업은 열만 바꾸면 되서 한 산업군 씩만 코드를 올리겠습니다)
basic_materials = merged_data[['Basic Materials']].reset_index()
basic_materials.columns = ['ds', 'y']

# 날짜 형식 변환
basic_materials['ds'] = pd.to_datetime(basic_materials['ds'])

# 2013년부터 2023년까지 학습용 데이터와 2024년 테스트 데이터로 분할
train_data = basic_materials[(basic_materials['ds'] >= '2013-01-01') & (basic_materials['ds'] < '2024-01-01')]
test_data = basic_materials[basic_materials['ds'].dt.year == 2024]

# Prophet 모델 생성 및 학습
model = Prophet()
model.fit(train_data)

# 미래 데이터 생성 (2024년 예측)
future = model.make_future_dataframe(periods=365)  # 2024년 전체 예측
forecast = model.predict(future)

# 예측 결과 시각화
fig = model.plot(forecast)
plt.title('Basic Materials 산업 예측 결과')
plt.xlabel('날짜')
plt.ylabel('주가')

# 실제값과 예측값 비교
plt.scatter(test_data['ds'], test_data['y'], color='red', label='Actual', s=10)  # 실제값
plt.plot(forecast['ds'], forecast['yhat'], color='blue', label='Predicted')  # 예측값
plt.legend()
plt.show()

# 예측값과 실제값 비교
comparison_df = test_data.merge(forecast[['ds', 'yhat']], on='ds', how='left')
comparison_df['error'] = comparison_df['yhat'] - comparison_df['y']

# 예측 성능 확인
print(comparison_df[['ds', 'y', 'yhat', 'error']])


#2013~2022 -> 2023 예측
basic_materials = merged_data[['Basic Materials']].reset_index()
basic_materials.columns = ['ds', 'y']

# 날짜 형식 변환
basic_materials['ds'] = pd.to_datetime(basic_materials['ds'])

# 2013년부터 2022년까지 학습용 데이터와 2023년 테스트 데이터로 분할
train_data = basic_materials[(basic_materials['ds'] >= '2013-01-01') & (basic_materials['ds'] < '2023-01-01')]
test_data = basic_materials[basic_materials['ds'].dt.year == 2023]

# Prophet 모델 생성 및 학습
model = Prophet(
    changepoint_prior_scale=0.1,  # 변화점을 좀 더 쉽게 반영
    seasonality_prior_scale=15    # 계절성을 좀 더 반영
)
model.fit(train_data)

# 미래 데이터 생성 (2023년 예측)
future = model.make_future_dataframe(periods=365)  # 2023년 전체 예측
forecast = model.predict(future)

# 예측 결과 시각화
fig = model.plot(forecast)
plt.title('Basic Materials 산업 예측 결과 (2023년)')
plt.xlabel('날짜')
plt.ylabel('주가')

# 실제값과 예측값 비교
plt.scatter(test_data['ds'], test_data['y'], color='red', label='Actual', s=10)  # 실제값
plt.plot(forecast['ds'], forecast['yhat'], color='blue', label='Predicted')  # 예측값
plt.legend()
plt.show()

# 예측값과 실제값 비교
comparison_df = test_data.merge(forecast[['ds', 'yhat']], on='ds', how='left')
comparison_df['error'] = comparison_df['yhat'] - comparison_df['y']

# 예측 성능 확인
print(comparison_df[['ds', 'y', 'yhat', 'error']])


#2022년 1년 학습 후 2023년 10일 예측
basic_materials = merged_data[['Basic Materials']].reset_index()
basic_materials.columns = ['ds', 'y']

# 날짜 형식 변환
basic_materials['ds'] = pd.to_datetime(basic_materials['ds'])

# 2022년 학습용 데이터와 2023년 10일 테스트 데이터로 분할
train_data = basic_materials[(basic_materials['ds'] >= '2022-01-01') & (basic_materials['ds'] < '2023-01-01')]
test_data = basic_materials[(basic_materials['ds'] >= '2023-01-01') & (basic_materials['ds'] < '2023-01-11')]

# Prophet 모델 생성 및 학습
model = Prophet(
    changepoint_prior_scale=0.1,  # 변화점을 좀 더 쉽게 반영
    seasonality_prior_scale=15    # 계절성을 좀 더 반영
)
model.fit(train_data)

# 미래 데이터 생성 (2023년 첫 10일 예측)
future = model.make_future_dataframe(periods=10)  # 2023년 첫 10일 예측
forecast = model.predict(future)

# 예측 결과 시각화
fig = model.plot(forecast)
plt.title('Basic Materials 산업 예측 결과 (2023년 첫 10일)')
plt.xlabel('날짜')
plt.ylabel('주가')

# 실제값과 예측값 비교
plt.scatter(test_data['ds'], test_data['y'], color='red', label='Actual', s=10)  # 실제값
plt.plot(forecast['ds'], forecast['yhat'], color='blue', label='Predicted')  # 예측값
plt.legend()
plt.show()

# 예측값과 실제값 비교
comparison_df = test_data.merge(forecast[['ds', 'yhat']], on='ds', how='left')
comparison_df['error'] = comparison_df['yhat'] - comparison_df['y']

# 예측 성능 확인
print(comparison_df[['ds', 'y', 'yhat', 'error']])