# 导入库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# 定义数据
x = np.array(
    [0.017419779, 0.01528953, 0.015247662, 0.017215299, 0.015771182, 0.012399369, 0.01876411, 0.023248221, 0.012331862,
     0.022859227,
     0.01200794, 0.010960993, 0.016427449, 0.011568659, 0.01190818, 0.008291182, 0.016740123, 0.01151162, 0.015197981,
     .009978794])
y = np.array([.062690648, .025508717, .054971039, .061990906, .056883899,
              0.044727502,
              0.067822799,
              0.084053483,
              0.044510084,
              0.082669722,
              0.043287052,
              0.039583238,
              0.059255931,
              0.041754687,
              0.042947577,
              0.029838217,
              0.030444084,
              0.041543512,
              0.054839499,
              0.036055089])

# 计算决定系数
r2 = r2_score(y_true=y, y_pred=x)  # 使用sklearn.metrics.r2_score函数

# 打印结果
print("The coefficient of determination is", r2)

# 绘制条形图
plt.bar(x="R-squared", height=r2, color="green", label="coefficient of determination")  # 条形图
plt.xlabel("R-squared")
plt.ylabel("Value")
plt.legend()
plt.show()
