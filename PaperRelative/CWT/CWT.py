import numpy as np
import pandas as pd
import pywt
import warnings

from matplotlib import MatplotlibDeprecationWarning

warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

datatest = pd.read_excel('../Resource/ExperimentDataBase/')
origin_var = datatest[['Band1蓝','Band2绿','Band3红','Band4红边','Band5近红外1','Band6近红外2','SPAD','株高','LAI','Gs','含水率']]
# 处理NaN值
origin_var = origin_var.fillna(method='ffill')  # 使用前一个非NaN值进行填充

# 处理inf值
origin_var = origin_var.replace([np.inf, -np.inf], np.nan)  # 将inf值替换为NaN值
origin_var = origin_var.fillna(method='ffill')  # 使用前一个非NaN值进行填充

datatest_arry = np.array(origin_var)
"""
连续小波变换(CWT)是一种数学工具，其作用是将时域信号转换为时间-频率分析的表示，以便更好地理解信号的特性和特征。CWT可用于信号处理、图像处理、模式识别、数据压缩、金融分析等领域。

CWT的输出是一个矩阵，其中每一行表示一个尺度(scale)的频谱，每一列表示一个时间点，可以用来展示信号在不同尺度下的频率成分随时间的变化情况。这使得CWT可以有效地检测和分离信号
中的不同频率成分，例如短时突发的信号或周期性的信号。

CWT有助于发现信号中的重要特征，例如局部极值、边缘、纹理等。这使得CWT成为许多信号处理和分析任务中的有用工具，例如语音识别、图像处理、生物医学工程等。

"""
# 生成一个简单的测试信号，包含三个正弦波
t = np.linspace(0, 1, 500, endpoint=False)
datatest_arry = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t) + np.sin(2*np.pi*30*t)

# 定义连续小波变换的参数
scales = np.arange(1, 100)
wavelet = pywt.ContinuousWavelet('mexh')

# 进行连续小波变换
coeffs, freqs = pywt.cwt(datatest_arry, scales, wavelet)

# 绘制连续小波变换系数的热力图
import matplotlib.pyplot as plt
plt.imshow(coeffs, cmap='coolwarm', aspect='auto', extent=[0, 1, 100, 1])
plt.xlabel('Time (s)')
plt.ylabel('Scale')
plt.colorbar()
plt.show()
