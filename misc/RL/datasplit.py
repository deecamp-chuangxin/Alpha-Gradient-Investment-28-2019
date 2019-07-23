import pandas as pd
import numpy as np
import pickle

#按照Tencent.csv的格式，把数据分成每日可交易时段的价格list。随便写的
#一个交易日大概330个数据（香港交易时间上午两个半小时下午三个小时）
file='/Users/liuzf/Documents/GitHub/DDRL_trading/Tencent/Tencent.csv'
with open(file) as f:
    lines = f.readlines()
origindata=[i.split(',') for i in lines]
data = pd.DataFrame(origindata)

idx = pd.IndexSlice
listA = list(data.loc[idx[:,0]])
tradedate = sorted(set(listA),key = listA.index)    #tradedate[0]要除去，是个系统文件
tradedate=tradedate[201:]  #前200天放到train，后面的352天放到test
for date in tradedate:  #按交易日分数据，每日的价格列表存成pckl文件
    print(date)
    pricelist=[]
    for data in origindata:
        if date==data[0]:
            datatime = data[1].split(':')
            if int(datatime[0])>8 and int(datatime[0])<17:
                if int(datatime[0])==9:
                    if int(datatime[1])>29:
                        pricelist.append((float(data[3])+float(data[4]))/2) #(bid+ask)/2当作交易价格
                elif int(datatime[0])==12:
                    if int(datatime[1])==0:
                        pricelist.append((float(data[3])+float(data[4]))/2)
                elif int(datatime[0])==16:
                    if int(datatime[1])==0:
                        pricelist.append((float(data[3])+float(data[4]))/2)
                else:
                    pricelist.append((float(data[3])+float(data[4]))/2)
    datesplit=date.split('/')
    year = datesplit[2]
    if int(datesplit[0])<10:
        month = '0'+datesplit[0]
    else:
            month = datesplit[0]
    if int(datesplit[1])<10:
        day = '0'+datesplit[1]
    else:
        day=datesplit[1]
    name = year+month+day
    vf1 = open('/Users/liuzf/Documents/GitHub/DDRL_trading/Tencent/test/'+name+'.pckl','wb')
    print(name)
    pickle.dump(pricelist,vf1)
    print('done')
    vf1.close()
