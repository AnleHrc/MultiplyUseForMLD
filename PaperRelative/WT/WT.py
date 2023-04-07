import numpy as np
import pandas as pd
import pywt

# 读取数据
df = pd.read_excel('../Resource/SPAD(1).xlsx', usecols=['blue', 'green', 'red', 'redbound', 'ni1', 'ni2'])

# 定义小波变换的函数
def wavelet_transform(data, wavelet='db4', level=1):
    # 对数据进行小波分解
    coeffs = pywt.wavedec(data, wavelet, level=level)
    # 重构小波分解得到的系数
    reconstructed_data = pywt.waverec(coeffs, wavelet)
    return reconstructed_data

# 对每一列数据进行小波变换
for col in df.columns:
    df[col+'_wt'] = wavelet_transform(df[col])

# 将结果保存到 Excel 文件
df.to_excel('WT_Output1.xlsx', index=False)
