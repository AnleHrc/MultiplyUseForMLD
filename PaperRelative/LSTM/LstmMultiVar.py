import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.losses import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from datetime import datetime

# 读取数据并转换为numpy数组
df = pd.read_csv('../Resource/Bloomsbury_clean_modify.csv')
df['date'] = pd.to_datetime(df['date'])
df['date'] = df['date'].astype('int64') // 10**9  # Convert to Unix timestamp
dataset = df.values

# 将数据集归一化
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# 划分训练集和测试集
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# 将数据集转换为特征和标签
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(dataset[i+look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# 构建LSTM模型
model = Sequential()
model.add(LSTM(4, input_shape=(look_back, dataset.shape[1])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=20, batch_size=1, verbose=2)

# 在训练集上进行预测
trainPredict = model.predict(trainX)
trainPredict = np.concatenate((trainPredict, np.zeros((trainPredict.shape[0], trainX.shape[1]-1))), axis=1)




# 在测试集上进行预测
testPredict = model.predict(testX)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])


# 绘制训练集预测结果图像
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# 绘制测试集预测结果图像
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2):len(dataset), :] = testPredict

# 绘制原始数据图像和预测结果图像
plt.plot(scaler.inverse_transform(dataset), label='True Data')
plt.plot(trainPredictPlot, label='Training Prediction')
plt.plot(testPredictPlot, label='Testing Prediction')
plt.legend()
plt.show()
