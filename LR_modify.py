#coding:utf-8
import pandas as pd
# from sklearn.preprocessing import OneHotEncoder
import warnings
import xgboost as xgb
import numpy as np
def delUnname0(df):
    df = df.drop('Unnamed: 0',axis=1)
    return df
def xgb_eval_custom_r(y_pred, dtrain):
    y_true = dtrain.get_label()
    up = np.sum(y_pred - y_true)
    down = np.sum(y_pred + y_true)
    whole = np.abs(up/down)
    Loss = whole/len(y_true)
    #here only predict one shop
    return 'Loss', Loss
#preprocessing normailze?No.Do a one hot encoding~~Because the data is so small~~
def TimeFeat(df):
    columns = df.columns
    df['daysinmonth'] = pd.Index(df[columns[1]]).daysinmonth
    df['weekofyear'] = pd.Index(df[columns[1]]).weekofyear
    df['dayofweek'] = pd.Index(df[columns[1]]).dayofweek
    dummies_daysinmonth = pd.get_dummies(df['daysinmonth'],prefix='daysinmonth')
    dummies_weekofyear = pd.get_dummies(df['weekofyear'],prefix='weekofyear')
    dummies_dayofweek = pd.get_dummies(df['dayofweek'],prefix='dayofweek')
    df_dummies = df.join([dummies_daysinmonth,dummies_weekofyear,dummies_dayofweek],how='outer')
    df_dummies = df_dummies.drop(['dayofweek','daysinmonth','weekofyear'],axis=1)
    return df_dummies
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
def GenerateTimeF(eachShop):
    eachShop.reset_index(inplace=True)
    eachShop.drop(['index'],inplace=True,axis=1)
    Yesterday_col = {}
    TwoDayAgo_col = {}
    TwoDayAgo_col = pd.DataFrame(TwoDayAgo_col)
    Yesterday_col = pd.DataFrame(Yesterday_col)
    Yesterday_col['yesterday'] = eachShop['Num']
    TwoDayAgo_col['TwoDayAgo_col'] = eachShop['Num']

    eachShop = eachShop.drop(0).drop(1)
    eachShop.reset_index(inplace=True)
    eachShop.drop(['index'],inplace=True,axis=1)

    Yesterday_col = Yesterday_col.drop(0)
    Yesterday_col.reset_index(inplace=True)
    Yesterday_col.drop(['index'],inplace=True,axis=1)

    TwoDayAgo_col.reset_index(inplace=True)
    TwoDayAgo_col.drop(['index'],inplace=True,axis=1)

    newdata = TwoDayAgo_col.join(Yesterday_col)
    newdata = newdata.join(eachShop)
    newdata.drop(['shop_id'],axis=1,inplace=True)
    newdata['difference_two_day'] = newdata['yesterday'] - newdata['TwoDayAgo_col']
    newdata['meanValue14'] =0
    newdata['meanValue7'] =0

    for x in range(14,len(newdata)-15):
        eachline = newdata.ix[x:x]
        fourteen = newdata.ix[x-14:x-1]
        meanValue14 = fourteen['TwoDayAgo_col'].mean()
        meanValue7 = fourteen.ix[len(fourteen)-7+x-14:len(fourteen)-1+x-14]
        meanValue7 = meanValue7['TwoDayAgo_col'].mean()
        eachline['meanValue14'] = meanValue14
        eachline['meanValue7'] = meanValue7
        newdata.update(eachline)
    newdata.drop(len(newdata)-1,inplace=True)
    newdata.drop(len(newdata)-1,inplace=True)
    for x in range(0,14):
        newdata.drop(x,inplace=True)

    newdata.reset_index(inplace=True)
    newdata.drop(['index'],axis=1,inplace=True)

    basePredict = newdata.ix[len(newdata)-14-13:len(newdata)-1]
    for x in range(0,14):
        newdata.drop(len(newdata)-1,inplace=True)
    return basePredict,newdata
def SplitData(PredictData,newdata):
    PredictData = PredictData.reset_index()
    PredictData.drop(['index'],axis=1,inplace=True)
    train_data = newdata.ix[0:int(0.9*len(newdata))]
    test_data = newdata.ix[int(0.9*len(newdata)):len(newdata)-1]
    return PredictData,train_data,test_data
def FittingModel(train_data,test_data,ToDrop,shop_id):
    train_xgb = xgb.DMatrix(data=train_data.drop(ToDrop,axis=1)
                            ,label=train_data['Num'])
    valid_xgb = xgb.DMatrix(data=test_data.drop(ToDrop,axis=1)
                            ,label=test_data['Num'])

    params = {
      'objective': 'reg:linear'
      ,'eta': 0.1
      ,'max_depth': 6
      , 'subsample': 0.4
      , 'colsample_bytree': 0.9
      ,'min_child_weight': 12
      ,'gamma': 0.07
      , 'seed': 10
    ,'reg_alpha': 0.06
    }
    evallist = [(train_xgb, 'train'), (valid_xgb, 'valid')]
    model = xgb.train(params.items()
                      , dtrain=train_xgb
                      , num_boost_round=10000
                      , evals=evallist
                      , early_stopping_rounds=20
                      , maximize=False
                      , verbose_eval=10
                      , feval=xgb_eval_custom_r
                  )
    print "now it is:",shop_id
    print ('get info from model')
    print (model.best_score, model.best_iteration,model.best_ntree_limit)
    return model
p_result = []
Customer_Flow = LoadData()
all_data = GenerateAllData(Customer_Flow)

for shop_id, eachShop in all_data.groupby(['shop_id']):
    warnings.filterwarnings("ignore")
    eachShop = TimeFeat(eachShop)

    PredictData, newdata = GenerateTimeF(eachShop)

    PredictData,train_data, test_data = SplitData(PredictData,newdata)
    ToDrop = ['Num', 'data']
    print "fitting data on model....."
    model =  FittingModel(train_data,test_data,ToDrop,shop_id)
    result = []
    dataLen = len(PredictData)

    print "predicting...."
    for x in range(13, dataLen):
        first = PredictData.ix[x:x]
        if x == 14:
            first['yesterday'] = firstNum
            Next_TwoDayAgo_col = firstNum
            fourteen = PredictData.ix[x-14:x-1]
            meanValue14 = fourteen['TwoDayAgo_col'].mean()
            meanValue7 = fourteen.ix[len(fourteen)-7+x-14:len(fourteen)-1+x-14]
            meanValue7 = meanValue7['TwoDayAgo_col'].mean()
            first['meanValue14'] = meanValue14
            first['meanValue7'] = meanValue7
            PredictData.update(first)
        if x >= 15:
            first['yesterday'] = firstNum
            first['TwoDayAgo_col'] = Next_TwoDayAgo_col
            Next_TwoDayAgo_col = firstNum
            fourteen = PredictData.ix[x-14:x-1]
            meanValue14 = fourteen['TwoDayAgo_col'].mean()
            meanValue7 = fourteen.ix[len(fourteen)-7+x-14:len(fourteen)-1+x-14]
            meanValue7 = meanValue7['TwoDayAgo_col'].mean()
            first['meanValue14'] = meanValue14
            first['meanValue7'] = meanValue7
            PredictData.update(first)

        predict_xgb = xgb.DMatrix(data=first.drop(ToDrop, axis=1))
        firstNum = model.predict(predict_xgb, ntree_limit=model.best_ntree_limit)
        result.append(firstNum)
    result = pd.DataFrame(result)
    result = result.rename(columns={0:'p'})
    each_line = []
    each_line.append(shop_id)
    for Num in result.p:
        each_line.append(Num)
    p_result.append(each_line)
    print " the task  of ",shop_id,"is finished!!"
p_result = abs(pd.DataFrame(p_result).astype(int))
p_result.to_csv('LR_modify.csv', header=False, index=False, encoding='utf-8')