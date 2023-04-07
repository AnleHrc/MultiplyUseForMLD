import matplotlib.pyplot as plt
import numpy as np
import warnings
import pandas as pd
from matplotlib import MatplotlibDeprecationWarning

warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

def SPA(data, threshold):
    """
    Sequential Projection Algorithm (SPA)
    连续投影算法 (Successive Projections Algorithm, SPA) 是一种特征选择算法，
    它的主要作用是从原始数据中选择出最相关的特征子集，
    以用于分类或回归等任务。它通过投影数据到一系列随机方向来计算特征的重要性，
    然后选择具有最高重要性的特征。在特征选择的过程中，SPA不需要任何预先假设或模型，
    只需要原始数据即可。它可以帮助提高模型的性能，同时也可以降低维度，减少冗余信息。

    Parameters:
        data (ndarray): 输入数据，每一列代表一个数据点
        threshold (float): 阈值，当投影误差小于该值时停止迭代

    Returns:
        ndarray: 迭代结束后重构的数据
    """
    # 投影向量
    proj_vec = np.zeros_like(data)
    # 投影误差
    proj_error = np.zeros(data.shape[1])
    # 重构数据
    rec = np.zeros_like(data)
    # 初始化投影向量为第一个数据点
    proj_vec[:, 0] = data[:, 0]
    for i in range(1, data.shape[1]):
        # 计算前i-1个向量的投影系数
        proj_data = np.dot(proj_vec[:, :i].T, data[:, i])
        # 计算投影误差
        proj_error[i] = np.linalg.norm(data[:, i] - np.dot(proj_vec[:, :i], proj_data))
        # 寻找最大投影误差的数据点
        max_error_index = np.argmax(proj_error[:i+1])
        # 如果最大误差小于阈值，停止迭代
        if proj_error[max_error_index] < threshold:
            break
        # 更新投影向量
        proj_vec[:, max_error_index] += data[:, i] - np.dot(proj_vec[:, max_error_index], data[:, i]) * proj_vec[:, max_error_index]
        # 归一化
        proj_vec[:, max_error_index] /= np.linalg.norm(proj_vec[:, max_error_index])
    # 重构数据
    for i in range(data.shape[1]):
        rec[:, i] = np.dot(proj_vec[:, :i+1], np.dot(proj_vec[:, :i+1].T, data[:, i].reshape(-1,1))).flatten()
    return rec


# ,'SPAD','株高','LAI','Gs','含水率'
# Read the Excel file
df = pd.read_excel('../Resource/ExperimentDataBase/',usecols=['Band1蓝','Band2绿','Band3红','Band4红边','Band5近红外1','Band6近红外2'])

# 处理NaN值
origin_var = df.fillna(method='ffill')  # 使用前一个非NaN值进行填充

# 处理inf值
origin_var = df.replace([np.inf, -np.inf], np.nan)  # 将inf值替换为NaN值
origin_var = df.fillna(method='ffill')  # 使用前一个非NaN值进行填充

# Convert the DataFrame to a numpy array
data = origin_var.values

# 设置阈值
threshold = 0.5

# 运行SPA算法
rec = SPA(data, threshold)

# 将重构后的数据保存到DataFrame对象中
df2 = pd.DataFrame(rec)

# 保存DataFrame对象到excel文件中
df2.to_excel('reconstructed_pad.xlsx',index=False)

# 绘制原始数据和重构数据的图像
plt.subplot(2, 1, 1)
plt.plot(data)
plt.title('Original Data')
plt.subplot(2, 1, 2)
plt.plot(rec)
plt.title('Reconstructed Data')
plt.show()


