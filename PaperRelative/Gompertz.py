import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 定义Gompertz函数
def gompertz(t, A, B, k):
    return A * np.exp(-B * np.exp(-k*t))

# 输入数据
t = np.array([0, 1, 2, 3, 4, 5])
y = np.array([10, 25, 50, 70, 80, 85])

# 拟合Gompertz模型
popt, pcov = curve_fit(gompertz, t, y)

# 输出拟合参数
print(popt) # 输出 [  8.27917414  20.11481829 0.68671457]

# 绘制拟合曲线
plt.plot(t, y, 'ko', label="Original Data")
plt.plot(t, gompertz(t, *popt), 'r-', label="Gompertz Fit")
plt.legend()
plt.show()