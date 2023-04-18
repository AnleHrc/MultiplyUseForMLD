import numpy as np
import pandas as pd
import scipy.io as sio
import seaborn as sns
import chardet
# def matlab_to_csv(data_m,rename):
#     mat_file = sio.loadmat('../Data/'+'{}'.format(data_m))
#
#     data_q2 = mat_file['Z2']
#     # 将数组转换为DataFrame
#     df_q2 = pd.DataFrame(data_q2)
#     # print(df_q2)
#     # 将DataFrame保存为CSV文件
#     # test_v = '{}'.format(rename) + '.csv'
#     # print(test_v)
#     df_q2.to_csv('{}'.format(rename) + '.csv', index=False, header=False)
#


    # 读取保存的QUBO矩阵
q_matrix = pd.read_csv('data_q2.csv')
q_matrix.head()

from matplotlib import pyplot as plt


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 绘制热图
sns.heatmap(q_matrix)
# 添加标题
plt.title('Q矩阵可视化热图')
plt.savefig('Q_matrix_two.png',dpi=1000)




# 绘制密度图
sns.kdeplot(q_matrix.values.flatten(), shade=True, bw=.15)
# 绘制等高线图
sns.kdeplot(q_matrix.values.flatten(), shade=False, n_levels=10, bw=.15)
plt.title('Q矩阵数据分布图')
plt.savefig('q_2_data_distribute.png')