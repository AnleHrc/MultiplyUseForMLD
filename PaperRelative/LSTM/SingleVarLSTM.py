import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 准备数据集
data = pd.read_csv('../Resource/Bloomsbury_clean_modify_copy2.csv')

# 将日期时间列转换为秒级时间戳
data['date'] = pd.to_datetime(data['date'])
data['date'] = data['date'].astype(np.int64) // 10**9

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.values)

# 将时间序列按照时间顺序划分为多个子序列
look_back = 2
X, Y = [], []
for i in range(len(scaled_data) - look_back - 1):
    X.append(scaled_data[i:(i + look_back), :])
    Y.append(scaled_data[i + look_back, :])
X = np.array(X)
Y = np.array(Y)

# 划分训练集和验证集
train_size = int(len(X) * 0.7)
X_train, X_val = X[:train_size], X[train_size:]
Y_train, Y_val = Y[:train_size], Y[train_size:]

# 构建LSTM模型
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, X.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(Y.shape[1]))
model.compile(loss='mean_squared_error', optimizer='adam')

# 训练模型
model.fit(X_train, Y_train, epochs=50, batch_size=64, validation_data=(X_val, Y_val), verbose=2)

# 预测未来天气状况
last_input = X[-1]
prediction = []
for i in range(7):
    last_input = last_input.reshape((1, look_back, X.shape[2]))
    output = model.predict(last_input)
    prediction.append(output[0])
    last_input = np.vstack((last_input[0][1:], output))

# 将预测结果反标准化处理
prediction = scaler.inverse_transform(prediction)

# 输出预测结果
# print(prediction)
# 绘制训练损失和验证损失
plt.plot(model.history.history['loss'], label='train_loss')
plt.plot(model.history.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 绘制真实值 和 预测值
plt.plot(data[-7:].index, data[-7:].values, label='actual')
plt.plot(data[-7:].index, prediction, label='prediction')
plt.title('Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()