import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def logistic_model(x,a,b,c):
    return c/(1+np.exp(-(x-b)/a))

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([0,0,0,0,0.2,0.4,0.6,0.8,1,1])

p0 = np.random.exponential(size=3)

popt, pcov = curve_fit(logistic_model, x, y, p0)

xx = np.linspace(0, 30, 200)
yy = logistic_model(xx, *popt)

plt.plot(xx,yy)
plt.scatter(x, y)

plt.show()