from sklearn.linear_model import LinearRegression
import pandas as pd

#Read File
df_X = pd.read_excel('Resource/SPAD(1).xlsx',usecols=['blue','green','red','redbound','ni1','ni2']);
df_y= pd.read_excel('Resource/SPAD(1).xlsx',usecols=['SPAD']);
# 创建线性回归模型对象
lr = LinearRegression()

# 训练模型，获取特征的权重值
lr.fit(df_X, df_y)
weights = lr.coef_

print(weights)
