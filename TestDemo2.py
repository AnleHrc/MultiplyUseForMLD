import numpy as np
import pandas as pd

# 变异系数发 计算 权重
datatest = pd.read_excel('PaperRelative/Resource/data3-28.xlsx')
var_origin = datatest[['SPAD','AGB','LAI','Ci']]

def sum_cv(data):
    sum_var = {}
    mean_val = np.mean(data,axis=0)
    std_val = np.std(data,axis=0)
    cv_val = (std_val / mean_val)
    for i,var_name in enumerate(data.columns):
        sum_var[var_name] =cv_val[i]
    return sum_var

cv_var = sum_cv(var_origin)
sum_vars = sum(cv_var.values())


def WeightOfCv(data):
    mean_val = np.mean(data, axis=0)
    std_val = np.std(data, axis=0)
    cv_val = (std_val / mean_val)
    weight = cv_val / sum_vars
    return weight

print(WeightOfCv(var_origin))




