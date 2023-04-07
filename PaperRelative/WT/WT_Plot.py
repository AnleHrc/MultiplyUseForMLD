import pandas as pd
import pywt
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", message="Support for FigureCanvases without a required_interactive_framework attribute was deprecated")

# 读取Excel文件中的数据
datatest = pd.read_excel('../Resource/ExperimentDataBase/')
origin_var = datatest[['Band1蓝','Band2绿','Band3红','Band4红边','Band5近红外1','Band6近红外2','SPAD','株高','LAI','Gs','含水率']]

# 提取需要处理的数据列和时间序列
y = origin_var['LAI'].values
# x = df['date'].values

# 对信号进行小波变换
coeffs = pywt.wavedec(y, 'db4', level=1)

# 将变换后的系数重构回原信号
y_reconstructed = pywt.waverec(coeffs, 'db4')

# 绘制原信号和重构后的信号
plt.plot(y, label='Original')
plt.plot(y_reconstructed, label='Reconstructed')
plt.legend()
plt.show()
# 将重构后的结果保存到Excel文件中
result = pd.DataFrame({'Reconstructed': y_reconstructed})

result.to_excel('WT_Reconstructed_result_1.xlsx', index=False)
