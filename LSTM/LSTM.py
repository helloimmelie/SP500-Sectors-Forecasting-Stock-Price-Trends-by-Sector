import pandas as pd 

from sklearn.preprocessing import MinMaxScaler

import numpy as np

import torch
import torch.nn as nn

import time

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error

import math, time


import plotly.express as px
import plotly.graph_objects as go

def map_index_from_other_df(df1, df2, column_name):
    # df2의 Symbol을 df1의 인덱스에 매핑
    index_map = df1
    print(index_map)
    df2['Index'] = df2[column_name].map(index_map)
    return df2

def split_data(stock, lookback):
    data_raw = stock # convert to numpy array
    data = []


    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback])
    
    data = np.array(data);
    test_set_size = int(np.round(0.2*data.shape[0]));
    train_set_size = data.shape[0] - (test_set_size);
    
    x_train = data[:train_set_size,:, : ]
    y_train = data[:train_set_size,:, : ]
    
    x_test = data[train_set_size:,:, : ]
    y_test = data[train_set_size:,:, : ]
    
    return [x_train, y_train, x_test, y_test]

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

#파라미터값 
input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs = 100

scaler = MinMaxScaler(feature_range=(-1, 1))

Industrial = ['Basic Materials',\
              'Communication Services', 'Consumer Cyclical'	,'Consumer Defensive'	,\
              'Energy'	,'Financial Services'	,'Healthcare'	,'Industrials	Real' ,'Estate	Technology',	'Utilities'] 

#결측치 삭제 해서 데이터 분석 진행 (결측치가 많긴 한데.. 일단 학습이 우선이라 뺴고 진행했음)
sector_grouped_data = pd.read_csv('./sector_grouped_data_stacked.csv')
listSector = list(sector_grouped_data['Sector'].unique())
#index.dropna()
sector_grouped_data= sector_grouped_data[sector_grouped_data['Date'] <= '2023-01-10' ]

sector_grouped_data['Adj Close'] = scaler.fit_transform(sector_grouped_data['Adj Close'].values.reshape(-1,1))

model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
criterion = torch.nn.MSELoss(reduction='mean')
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)


for sector in listSector:
    print(f'{sector} is training')
    sector_df = sector_grouped_data[sector_grouped_data['Sector'] == sector]

    lookback = 20 # choose sequence length
    
    
    #x_train, y_train, x_test, y_test = split_data(sector_df, lookback)

    #기간별로 어떻게든 쪼개기 
    x_train = sector_df[(sector_df['Date'] <= '2022-12-31') & (sector_df['Date'] >='2022-01-01')]['Adj Close']
    y_train = sector_df[(sector_df['Date'] <= '2022-12-31') & (sector_df['Date'] >='2022-01-01')]['Adj Close']
    x_test = sector_df[sector_df['Date'] > '2022-12-31']['Adj Close']
    y_test = sector_df[sector_df['Date'] > '2022-12-31']['Adj Close']


    #한번 
    #sector_df = sector_df.to_numpy().reshape((len(sector_df),1,1))

    x_train = torch.from_numpy(x_train.to_numpy().reshape((len(x_train),1,1))).type(torch.Tensor)
    x_test = torch.from_numpy(x_test.to_numpy().reshape(len(x_test),1,1)).type(torch.Tensor)
    y_train_lstm = torch.from_numpy(y_train.to_numpy().reshape(len(y_train),1,1)).type(torch.Tensor)
    y_test_lstm = torch.from_numpy(y_test.to_numpy().reshape(len(y_test),1,1)).type(torch.Tensor)

    hist = np.zeros(num_epochs)
    start_time = time.time()
    lstm = []
    epoch_num = 0
    checkpoint_path = f'model_epoch_{sector}_{epoch_num}.pt'

    for t in range(num_epochs):
        y_train_pred = model(x_train)

        loss = criterion(y_train_pred, y_train_lstm)
        print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if num_epochs%10==0:
            torch.save({'epoch':t,'model_state_dict':model.state_dict(),'optimizer_state_dict':optimiser.state_dict(),'loss':loss},checkpoint_path)
    
    
        training_time = time.time()-start_time
        print("Training time: {}".format(training_time))

    predict = pd.DataFrame(scaler.inverse_transform(y_train_pred.detach().numpy()))
    original = pd.DataFrame(scaler.inverse_transform(y_train_lstm.squeeze(1).detach().numpy()))

    # invert predictions
    y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
    y_train = scaler.inverse_transform(y_train_lstm.squeeze(1).detach().numpy())


    # make predictions
    y_test_pred = model(x_test)

    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    y_test = scaler.inverse_transform(y_test_lstm.squeeze(1).detach().numpy())

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))

    lstm.append(trainScore)
    lstm.append(testScore)
    lstm.append(training_time)


    
    # # shift train predictions for plotting
    # trainPredictPlot = np.empty_like(sector_grouped_data[sector_grouped_data['Sector'] == sector])
    # trainPredictPlot[:, :] = np.nan
    # trainPredictPlot[lookback:len(y_train_pred)+lookback, :] = y_train_pred

    # # shift test predictions for plotting
    # testPredictPlot = np.empty_like(sector_grouped_data[sector_grouped_data['Sector'] == sector])
    # testPredictPlot[:, :] = np.nan
    # testPredictPlot[len(y_train_pred)+lookback-1:len(sector_grouped_data[sector_grouped_data['Sector'] == sector])-1, :] = y_test_pred

    # adj_Close = pd.DataFrame(sector_grouped_data[sector_grouped_data['Sector'] == sector][['Adj Close']])
    # original= scaler.inverse_transform(adj_Close.values.reshape(-1,1))


    # predictions = np.append(trainPredictPlot, testPredictPlot, axis=1)
    # predictions = np.append(predictions, original, axis=1)

    # result = pd.DataFrame(predictions)

    plt.clf()

    # RMSE 계산
    trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
    testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1행 2열의 서브플롯 생성
    
    axes[0].cla()
    axes[1].cla()

    # Train 데이터 시각화
    axes[0].plot([val[0] for val in y_train], label='True Values (Train)', marker='o', linestyle='-', color='b')
    axes[0].plot([val[0] for val in y_train_pred], label='Predicted Values (Train)', marker='x', linestyle='--', color='r')
    axes[0].set_title(f'Train Data (RMSE: {trainScore:.2f})')
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('Values')
    axes[0].legend(loc='best')

    # Test 데이터 시각화
    axes[1].plot([val[0] for val in y_test], label='True Values (Test)', marker='o', linestyle='-', color='b')
    axes[1].plot([val[0] for val in y_test_pred], label='Predicted Values (Test)', marker='x', linestyle='--', color='r')
    axes[1].set_title(f'Test Data (RMSE: {testScore:.2f})')
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('Values')
    axes[1].legend(loc='best')

    # 그래프 출력
    plt.tight_layout()
    #plt.show()


    # fig = go.Figure()
    # fig.add_trace(go.Scatter(go.Scatter(x=result.index, y=result[0],
    #             mode='lines',
    #             name='Train prediction')))
    # fig.add_trace(go.Scatter(x=result.index, y=result[1],
    #             mode='lines',
    #             name='Test prediction'))
    # fig.add_trace(go.Scatter(go.Scatter(x=result.index, y=result[2],
    #             mode='lines',
    #             name='Actual Value')))
    # fig.update_layout(
    #     xaxis=dict(
    #     showline=True,
    #     showgrid=True,
    #     showticklabels=False,
    #     linecolor='white',
    #     linewidth=2
    # ),
    # yaxis=dict(
    #     title_text='Close (USD)',
    #     titlefont=dict(
    #     family='Rockwell',
    #     size=12,
    #     color='white',
    #     ),
    #     showline=True,
    #     showgrid=True,
    #     showticklabels=True,
    #     linecolor='white',
    #     linewidth=2,
    #     ticks='outside',
    #     tickfont=dict(
    #     family='Rockwell',
    #     size=12,
    #     color='white',
    #     ),
    # ),
    # showlegend=True,
    # template = 'plotly_dark'

    # )

    # annotations = []
    # annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
    #                         xanchor='left', yanchor='bottom',
    #                         text=f'Results (LSTM){sector}',
    #                         font=dict(family='Rockwell',
    #                                 size=26,
    #                                 color='white'),
    #                         showarrow=False))
    # fig.update_layout(annotations=annotations)

    plt.savefig(f"{sector}_result.png") 
    