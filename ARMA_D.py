#coding:utf-8
import warnings

import pandas as pd
import numpy as np
import statsmodels.api as sm
import sys
from statsmodels.tsa.stattools import adfuller

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


Customer_Flow = pd.read_csv("C:\Users\Administrator\PycharmProjects\Customer\Customer_Flow.csv")
Customer_Flow = delUnname0(Customer_Flow)
Customer_Flow['data'] = pd.to_datetime(Customer_Flow['data'])
p_result = []
errorshop = []
for shop_id, eachShop in Customer_Flow.groupby(['shop_id']):
    try:
        eachShop.index = eachShop['data']
        eachShop = eachShop.drop(['data', 'shop_id'], axis=1)

        eachShop_series = pd.Series(index=eachShop.index, data=eachShop.Num)

        # 数据预处理 不一定是取log,可能差分呢,也可能小波,也有可能卡尔曼傅里叶,也可以把序列进行分解然后进行拟合
        ts_log = np.log(eachShop_series)
        # 这里做一阶差分
        ts_log_diff = ts_log - ts_log.shift()
        ts_log_diff.dropna(inplace=True)

        warnings.filterwarnings("ignore")
        #  这里使用模型参数自动识别
        init_bic, init_p, init_q, init_properModel = proper_model(ts_log_diff, 10)
        print 'shop_id', shop_id, 'bic:', init_bic, 'p:', init_p, 'q:', init_q

        # 预测结果还原
        predict_ts = init_properModel.predict(start="2016-10-31", end="2016-11-14")
        predict_ts_cumsum = predict_ts.cumsum()

        # 把最后一个值作为基本值。或者拿14天做一个均值。作为基本值
        # 拿均值作为基本值不靠谱，因为还没有过滤异常值
        # base_value = sum(ts_log.ix[len(ts_log) - 14:len(ts_log) - 1]) / 14
        base_value = ts_log.ix[len(ts_log)-1]
        print "base value is :", base_value
        base_value_set = []
        for x in range(1, len(predict_ts) + 1):
            base_value_set.append(base_value)

        predictions_ARIMA_log = pd.Series(base_value_set, index=predict_ts.index)
        predictions_ARIMA_log.rename(columns={0: 'Num'}, inplace=True)
        predictions_ARIMA_log = predictions_ARIMA_log.add(predict_ts_cumsum)

        ts_log = ts_log.ix[predictions_ARIMA_log.index]
        rmse = np.sqrt(np.sum((predictions_ARIMA_log - ts_log) ** 2) / ts_log.size)
        print rmse

        log_recover = np.exp(predictions_ARIMA_log)
        log_recover.dropna(inplace=True)
        result = pd.DataFrame(log_recover)
        result = result.rename(columns={0: 'Num'})
        each_line = []
        each_line.append(shop_id)
        for x in result.Num:
            each_line.append(x)
        p_result.append(each_line)

    except Exception, e:
        errorshop.append(shop_id)
        print '------------------------------ValueError:note down error shop id--------------------------'
        continue
#      errorshop [1629.0, 1690.0, 1707.0, 1824.0, 1862.0]
print errorshop
p_result = abs(pd.DataFrame(p_result).astype(int))
p_result.to_csv('ARMA_D.csv',header=False,index=False,encoding='utf-8')