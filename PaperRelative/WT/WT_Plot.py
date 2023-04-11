import pandas as pd
import pywt
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", message="Support for FigureCanvases without a required_interactive_framework attribute was deprecated")

# 读取Excel文件中的数据
datatest = pd.read_excel('../Resource/ExperimentDataBase/CV1(1).xlsx')
origin_var = datatest[['SAVI','EVI','TCART','CIrededge１','DVI','OSAVI','RDVI','TVI','MNVI','MSAVI','MTVI','CV']]

# 提取需要处理的数据列和时间序列
# y = origin_var['LAI'].values
# x = df['date'].values

# 对信号进行小波变换
coeffs = pywt.wavedec(origin_var, 'db4', level=20)

# 将变换后的系数重构回原信号
y_reconstructed = pywt.waverec(coeffs, 'db4')

# 绘制原信号和重构后的信号
plt.plot(y_reconstructed, label='Original')
plt.plot(y_reconstructed, label='Reconstructed')
plt.legend()
plt.show()

print(y_reconstructed)
# # 将重构后的结果保存到Excel文件中
# result = pd.DataFrame({'Reconstructed': y_reconstructed})
#
# result.to_excel('WT_Reconstructed_result_1.xlsx', index=False)
