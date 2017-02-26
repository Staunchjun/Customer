import pandas as pd
import numpy as np
def delUnname0(df):
    df = df.drop('Unnamed: 0',axis=1)
    return df
def LoadData():
    pd.set_option('display.max_columns', 120)
    Customer_Flow = pd.read_csv("Customer_Flow.csv")
    Customer_Flow = delUnname0(Customer_Flow)
    Customer_Flow['data'] = pd.to_datetime(Customer_Flow['data'])
    return  Customer_Flow
def GenerateAllData(Customer_Flow):
    p_startt = pd.to_datetime("2016-11-01")
    p_endt = pd.to_datetime("2016-11-14")
    p_time = pd.date_range(start=p_startt, end=p_endt, freq='D')
    p_shop_id = range(1, 2001, 1)
    shopid = []
    p_t = []
    for shop_id in p_shop_id:
        for t in p_time:
            shopid.append(shop_id)
            p_t.append(t)
    predict = {'data': p_t, 'shop_id': shopid}
    predict = pd.DataFrame(predict)
    all_data = Customer_Flow.merge(predict, how='outer')
    return all_data
p_result = []
Customer_Flow = LoadData()
all_data = GenerateAllData(Customer_Flow)
p_result = []
for shop_id,eachShop in all_data.groupby(['shop_id']):
    eachShop.reset_index(inplace=True)
    eachShop.drop(['index'],inplace=True,axis=1)
    print 'computing....'
    result=[]
    result.append(shop_id)
    for x in range(len(eachShop)-1-13,len(eachShop)):
        eachline = eachShop.ix[x:x]
        fourteen = eachShop.ix[x-7:x-1]
        meanValue7 = fourteen['Num'].mean()
        eachline['Num'] = meanValue7
        result.append(meanValue7)
        eachShop.update(eachline)
    p_result.append(result)
    print shop_id,' prediction has finished!'
p_result = abs(pd.DataFrame(p_result).astype(int))
p_result.to_csv('meanValue7.csv', header=False, index=False, encoding='utf-8')