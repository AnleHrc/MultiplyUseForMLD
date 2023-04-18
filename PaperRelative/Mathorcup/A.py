import pandas as pd
import numpy as np
data_origin=pd.read_csv('../Mathorcup/Data/data_100.csv')
data=data_origin.values
#重塑形状
data=data.reshape((10,100,2))
change_value=0

#1.
'''card_id=0
yuzhi=0
for i in range(100):
    for j in range(10):
        temp=80000*a[j,i,0]-1080000*a[j,i,0]*a[j,i,1]
        if temp>z:
            z=temp
            card_id=i+1 #评分卡index
            yuzhi=j+1 #阈值
print('最大最终收入：',z,'评分卡：',card_id,'阈值：',yuzhi)'''
#2.
Result=[0,0,0]
for i in range(10):
    for j in range(10):
        for k in range(10):
            t=data[i,0,0]*data[j,1,0]*data[k,2,0]
            h=(data[i,0,1]+data[j,1,1]+data[k,2,1])/3
            temp=80000*t-1080000*t*h
            if temp>change_value:
                change_value=temp
                Result=[i+1,j+1,k+1]
                print(change_value)
                print(i+1,j+1,k+1)
print('MaxProfit:',change_value,'阈值组合：',Result)
print(change_value,Result)

# #3.
# card_id=[0,0,0]
# Result=[0,0,0]
# def get_Result_list(card_id):
#     z=0
#     for i in range(10):
#         for j in range(10):
#             for k in range(10):
#                 t=a[i,card_id[0],0]*a[j,card_id[1],0]*a[k,card_id[2],0]
#                 h=(a[i,card_id[0],1]+a[j,card_id[1],1]+a[k,card_id[2],1])/3
#                 temp=80000*t-1080000*t*h
#                 if temp>z:
#                     z=temp
#                     yuzhi=[i+1,j+1,k+1]
#     return z,yuzhi
# for i in range(98):
#     for j in range(i+1,99):
#         for k in range(j+1,100):
#             temp,yuzhi_temp=get_yuzhi_list([i,j,k])
#             if temp>z:
#                 z=temp
#                 yuzhi=yuzhi_temp
#                 card_id=[i+1,j+1,k+1]
# print('最大最终收入：',z,'评分卡组合：',card_id,'阈值组合：',yuzhi)
