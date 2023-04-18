import neal
import pandas as pd
from decimal import Decimal
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pyqubo import Array, Placeholder
from scipy.spatial.distance import cdist
from pyqubo import Binary, Constraint
import random
import re
import optuna
np.set_printoptions(formatter={'float':'{:.3f}'.format})
random.seed(10)

Data = pd.read_csv('../Data/data_100.csv')
m = 1000000 #总资金
p = 0.08 #利息率

def fun(lis):
    return [[lis[i],lis[i+1]] for i in range(0,200,2)]

data = Data.copy()
data['new']=data.apply(fun,axis=1)
# print(data['new'])
data=pd.DataFrame(data['new'].to_list(),columns=[f'Card{i}' for i in range(1,101)])
data.index=data.index+1
# print(data)

Res = np.zeros((100,10),dtype=float)
# print(Res)
s=0
for i in range(0,100):
    # print(data.iloc[:,i])
    k=0
    for x in data.iloc[:,i]:
        ans = Decimal(m*x[0]*(p-(1+p)*x[1])).quantize(Decimal('0.000'))
        # print(ans)
        Res[i][k] = ans
        if ans >0:
            s+=ans
        k+=1
# print(s)
min_r=np.min(Res)
max_r=np.max(Res)
# for i in range(100):
#     for j in range(10):
#         Res[i][j] = (Res[i][j] - min_r)/(max_r-min_r)
# print(Res)
'''QUBO建模过程'''
#100 张银行卡的 10 种阈值 取值为 0,1
X = np.array(Array.create("X", shape = (100,10), vartype = "BINARY"))
# print(X)

'''哈密顿量'''
# print(np.sum(X*Res))
# x1, x2,x3 ,x4,x5,x6,x7= Binary('x1'), Binary('x2'),Binary('x3'), Binary('x4'),Binary('x5'),Binary('x6'),Binary('x7')
H = -(np.sum(X*Res))+100000*Constraint(((1-np.sum(X))**2),label='1')
def getKey(dic,value):
    s=[]
    for key in dic:
        if dic[key] == value:
            # print(key)
            s.append(key)
        # re.sub(r'X/[/]', '', key)
    return s
def anser(s):
    l=getKey(s,1)
    # print(l)
    q = np.zeros((100, 10), dtype=float)
    for res in l:
        # print(res)
        res = re.sub(r'[X\[\]]','',res)
        # print(res)
    #     # print(len(res))
        if len(res)==3:
            a=int(res[0:2])
            b=int(res[2])
        else:
            a=int(res[0])
            b=int(res[1])
        print(a,b)
        q[a][b]=1
    h = np.sum(q*Res)
    return h
# for i in range(10):
model = H.compile()
qubo, offset = model.to_qubo()
# print(qubo)
max=0
sampler = neal.SimulatedAnnealingSampler()
bqm = model.to_qubo()
sampler = neal.SimulatedAnnealingSampler()
raw_solution = sampler.sample_qubo(qubo,num_reads=150)
# print(raw_solut   ion.first.sample)
h=anser(raw_solution.first.sample)
print(h)