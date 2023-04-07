import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 定义归一化函数
def normalize(data):
    max_val = np.max(data)
    min_val = np.min(data)
    norm_data = (data - min_val) / (max_val - min_val)
    return norm_data

# 定义计算变异系数的函数
def cv(data):
    mean_val = np.mean(data)
    std_val = np.std(data)
    cv_val = (std_val / mean_val) * 100
    return cv_val

# 定义数据集
datatest = pd.read_csv('./Resource/Bloomsbury_clean.csv')

# 归一化数据集
norm_data = normalize(datatest)

# 计算变异系数
cv_val = cv(norm_data)

print("归一化后的数据：", norm_data)
print("变异系数为：", cv_val)


# # 绘制直方图
# plt.hist(norm_data)
# plt.title("Normalized Data Histogram")
# plt.xlabel("Normalized Value")
# plt.ylabel("Frequency")
# plt.show()
