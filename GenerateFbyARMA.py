#coding:utf-8
import pandas as pd
import numpy as np
import statsmodels.api as sm
import sys
from statsmodels.tsa.stattools import adfuller
import warnings
def delUnname0(df):
    df = df.drop('Unnamed: 0', axis=1)
    return df
def proper_model(data_ts, maxLag):
    init_bic = sys.maxint
    init_p = 0
    init_q = 0
    init_properModel = None
    for p in np.arange(maxLag):
        for q in np.arange(maxLag):
            model = sm.tsa.ARMA(data_ts, order=(p, q), freq='D')
            try:
                results_ARMA = model.fit(disp=-1, method='css')
            except:
                continue
            bic = results_ARMA.bic
            if bic < init_bic:
                init_p = p
                init_q = q
                init_properModel = results_ARMA
                init_bic = bic
    print 'shop_id', shop_id, 'bic:', init_bic, 'p:', init_p, 'q:', init_q
    return init_bic, init_p, init_q, init_properModel
def test_DF(timeseries):
    # Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'

    a_timeseries = np.array(timeseries)
    dim_1_data = []
    for x in a_timeseries:
        dim_1_data.extend(x)

    dftest = adfuller(dim_1_data, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print dfoutput
def GenerateBaseValue(ts_log,period,train_f):
    # base_value = sum(ts_log.ix[len(ts_log) - period - 31:len(ts_log) - period]) / 31
    base_value = ts_log.ix[len(ts_log)-period]
    print "base value is :", base_value
    base_value_set = []
    for x in range(1, len(train_f) + 1):
        base_value_set.append(base_value)
    return base_value_set
def GetTimePeriod(data, period):
    data_len = len(data)
    last_data = data.ix[data_len - 1]
    start_data = data.ix[data_len - 1-period]
    PeriodData = data.ix[(data_len - 1-period):data_len - 1]
    end_date = last_data['data']
    start_date = start_data['data']
    print  "this is a start date", start_date
    print "this is a last date", end_date
    return start_date,end_date,PeriodData
def LoadData():
    Customer_Flow = pd.read_csv("C:\Users\Administrator\PycharmProjects\Customer\Customer_Flow.csv")
    Customer_Flow = delUnname0(Customer_Flow)
    Customer_Flow['data'] = pd.to_datetime(Customer_Flow['data'])
    return Customer_Flow
def OneStepShift(ts_log):
    ts_log_diff = ts_log - ts_log.shift()
    ts_log_diff.dropna(inplace=True)
    return ts_log_diff
def PreProcessing(eachShop):
    eachShop.index = eachShop['data']
    eachShop = eachShop.drop(['data', 'shop_id', 'index'], axis=1)
    return eachShop
# 这里的period表明拿多长时间前的前2周做均值
def ShiftRevert(train_f,period,ts_log):
    train_f_cumsum = train_f.cumsum()
    # 把最后一个值作为基本值。或者拿14天做一个均值。作为基本值
    base_value_set = GenerateBaseValue(ts_log,period,train_f)
    predictions_ARIMA_log = pd.Series(base_value_set, index=train_f.index)
    predictions_ARIMA_log.rename(columns={0: 'Num'}, inplace=True)
    predictions_ARIMA_log = predictions_ARIMA_log.add(train_f_cumsum)
    return predictions_ARIMA_log
def CalRmse(ts_log,predictions_ARIMA_log):
    ts_log = ts_log.ix[predictions_ARIMA_log.index]
    rmse = np.sqrt(np.sum((predictions_ARIMA_log - ts_log) ** 2) / ts_log.size)
    print rmse
def GenerateResult(predictions_ARIMA_log,NewCol):
    log_recover = np.exp(predictions_ARIMA_log)
    log_recover.dropna(inplace=True)
    result = pd.DataFrame(log_recover)
    result = result.rename(columns={0: NewCol})
    return result
Customer_Flow = LoadData()
p_result = []
errorshop = []
for shop_id, eachShop in Customer_Flow.groupby(['shop_id']):
    try:

        # 取出最后一天的日期，以及前31天的日期
        eachShop.reset_index(inplace='True')
        start_date,end_date,PeriodData = GetTimePeriod(eachShop,31)
        # 数据预处理
        eachShop_series = PreProcessing(eachShop)
        eachShop_series = pd.Series(index=eachShop.index, data=eachShop.Num)
        PeriodData = PreProcessing(PeriodData)

        # 数据预处理 不一定是取log,可能差分呢,也可能小波,也有可能卡尔曼傅里叶,也可以把序列进行分解然后进行拟合
        ts_log = np.log(eachShop_series)
        # 这里做一阶差分
        ts_log_diff = OneStepShift(ts_log)
        # 忽略掉警告
        warnings.filterwarnings("ignore")
        #  这里使用模型参数自动识别
        print "the model is fitting data......"
        init_bic, init_p, init_q, ARMA_D_model = proper_model(ts_log_diff, 10)
        init_bic2, init_p2, init_q2, ARMA_model = proper_model(ts_log, 10)
        print "the model is predicting data......"
        #训练数据的特征产生
        train_f_D = ARMA_D_model.predict(start=start_date, end=end_date)
        train_f = ARMA_model.predict(start=start_date, end=end_date)
        # 预测数据还原
        predictions_ARIMA_log = ShiftRevert(train_f_D, 31, ts_log)

        # rmse计算
        rmse_D = CalRmse(ts_log, predictions_ARIMA_log)
        rmse =CalRmse(ts_log,train_f)

        result_D = GenerateResult(predictions_ARIMA_log,'ARMA_Num_D')
        result = GenerateResult(train_f,'ARMA_Num')

        print "the model is generating feature......"
        PeriodData.index = result.index
        TrainData = PeriodData.join(result,how='outer')
        TrainData = TrainData.join(result_D,how='outer')
        TrainData = pd.DataFrame(TrainData)
        TrainData['shop_id'] = shop_id
        p_result.append(TrainData)
        print shop_id,'has finished all task!!!'
    except Exception, e:
        errorshop.append(shop_id)
        print '------------------------------ValueError:note down error shop id--------------------------'
        continue
base = p_result[0]
for x in range(len(p_result)):
    base = base.merge(p_result[x],how='outer')
base.to_csv('ARMA_New_Data.csv',index=False,encoding='utf-8')
print errorshop