
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import torch
def GetFileList(data_dir):
    train_dir = os.path.join(data_dir,'train')
    test_dir = os.path.join(data_dir,'test')
    
    train_files = os.listdir(train_dir)
    train_files=[i for i in train_files if i[0]!='.'] #排除隐藏文件干扰
    test_files = os.listdir(test_dir)
    test_files=[i for i in test_files if i[0]!='.'] #排除隐藏文件干扰
    train_files.sort()
    test_files.sort()
    
    N = len(train_files)
    N_train = int(N)
    
    return train_files[:N_train],test_files
#一个交易日大概330个数据（香港交易时间上午两个半小时下午三个小时）
data_dir = '/Users/liuzf/Documents/DDRL_trading/Tencent'
train_files, test_files = GetFileList(data_dir) 

allprice=[]
for file in train_files:
    vf = open(data_dir+'/train/'+file,'rb')
    price_list = pickle.load(vf)
    allprice.extend(price_list[45:])
    vf.close()

train2test = len(allprice)

for file in test_files:
    vf = open(data_dir+'/test/'+file,'rb')
    price_list = pickle.load(vf)
    allprice.extend(price_list[45:])
    vf.close()
print(0)
test_ut_list = torch.load('/Users/liuzf/Documents/DDRL_trading/result/TP_DDR30_4')
print(test_ut_list)
test_ut_list = list(test_ut_list)

xlist=range(len(allprice))
print(5)
plt.figure(figsize=(30,10))
print(1)
plt.plot(xlist,np.array(allprice)-allprice[0],'b')
print(len(test_ut_list))
plt.plot([i+train2test for i in range(len(test_ut_list))],test_ut_list,'r')
print(2)
plt.scatter(train2test, allprice[train2test]-allprice[0], s = 300, color = 'g')
print(3)
#plt.show()
plt.savefig('price1_vs2.png')

#读取train+test文件列表
#依次读取所有价格指标
#依次读取所有TP
#作图