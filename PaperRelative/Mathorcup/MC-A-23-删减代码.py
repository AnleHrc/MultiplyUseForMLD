import pandas as pd
import numpy as np
import itertools

data_origin = pd.read_pickle('Data/data_a2.pkl')
data_origin.head(1)

# 变量
m = 1000000
p = 0.08

# def cal_w2(lis): # 计算三张卡组合起来的利润,x是[t1,t2,t3],y是[h1,h2,h3]
#     x = lis[3]*lis[4]*lis[5]
#     y = np.mean(lis[6:])
#     return m*x*(p-(1+p)*y)

def cal_third_question(data_list): # 计算三张卡组合起来的利润,x是[t1,t2,t3],y是[h1,h2,h3]
    x = data_list[-6]*data_list[-5]*data_list[-4]
    y = np.mean(data_list[-3:])
    return m*x*(p-(1+p)*y)

# # 2
# Data2 = []
# for i, j, k in itertools.product(range(10), range(10), range(10)):
#     t1,h1 = df.iloc[i,0]
#     t2,h2 = df.iloc[j,1]
#     t3,h3 = df.iloc[k,2]
#     Data2.append([i,j,k,t1,t2,t3,h1,h2,h3])
#
# res2 = pd.DataFrame(Data2,columns=['Threshold1','Threshold2','Threshold3','t1','t2','t3','h1','h2','h3'])
# res2['w'] = res2.apply(cal_w2,axis=1)
# res2 = res2.sort_values('w',ascending=0)
# res2.head()

# 3
data = data_origin.copy()
data = data.applymap(lambda x:np.array(x))
data = data.to_numpy()

# 计算第三问
Result = []

for c1 in range(100): # 选第一张卡
    for c2 in range(c1+1,100): # 选第二张卡
        for c3, i, j, k in itertools.product(range(c2+1,100), range(10), range(10), range(10)):
            t1,h1 = data[i,c1]
            t2,h2 = data[j,c2]
            t3,h3 = data[k,c3]
            Result.append([c1,c2,c3,i,j,k,t1,t2,t3,h1,h2,h3])
            # break
        # break
    # break
res = pd.DataFrame(Result,columns= ['Card1','Card2','Card3','Threshold1','Threshold2','Threshold3','t1','t2','t3','h1','h2','h3'])
res['w'] = res.apply(cal_third_question,axis=1)
res = res.sort_values('w',ascending=0)
res.head()