from curses.ascii import NUL
from pyexpat import model
from statistics import median
import sys
import datetime, jpholiday
from webbrowser import get
import numpy as np
from sktime.datasets import load_airline
from sktime.forecasting.compose import make_reduction
from sklearn.ensemble import RandomForestRegressor
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
import pandas as pd
import matplotlib.pyplot as plt
# from sktime.forecasting.all import *


# input data
y = load_airline()
df_old = pd.read_excel('current_data.xlsx')
df_true = pd.read_excel('true_data.xlsx')
df_true['ds'] = pd.to_datetime(df_true['ds'])


def alg_1():
    # split data
    y_train, y_test = temporal_train_test_split(df_true)

    # 学習
    regressor = RandomForestRegressor()
    forecaster = make_reduction(
        regressor,
        strategy="recursive",
        window_length=12,
        scitype="infer",
    )
    forecaster.fit(y_train)

    #推論
    fh = np.arange(1, len(y_test) + 1)  # forecasting horizon
    y_pred = forecaster.predict(fh)

    # 評価
    mape = MeanAbsolutePercentageError()
    print(f"mape = {mape(y_test, y_pred)}")

    return

def alg_2():
    train, test = temporal_train_test_split(df_true, test_size=365)

    # 日時データをindexに変換
    train_sk = train.set_index("ds")
    test_sk = test.set_index("ds")

    # 学習
    TIMESPAN = 90
    model = ThetaForecaster(sp=TIMESPAN)
    model.fit(train["y"])

    fh = ForecastingHorizon(test.index, is_relative=False)

    # 予測データの作成
    pred_sktime = model.predict(fh)
    
    return

def alg_3():
    from neuralprophet import NeuralProphet
    from sklearn.metrics import mean_absolute_error
    import time
    
    st_time = time.time()
    
    #データセットの分割
    data_size = len(df_true)
    test_length = 10

    train = df_true[:-test_length]
    test = df_true[-test_length:]
    
    #機械学習モデルのインスタンス作成
    m = NeuralProphet(seasonality_mode='multiplicative')
    
    #validationの設定(3:1)&モデルのトレーニング
    metrics = m.fit(train, freq="D")

    #作ったモデルで予測
    future = m.make_future_dataframe(train, periods=test_length)
    forecast = m.predict(future)

    #予測結果をtestのデータフレームに追加
    test["pred"] = forecast["yhat1"].to_list()
    print(forecast.columns)
    
    #MAEで精度を確認
    print('MAE(NeuralProphet):')
    print(mean_absolute_error(test['y'], test['pred']))

    print(f"time: {time.time() - st_time}")

    #可視化 
    test.plot(title='Forecast evaluation')
    fig_comp = m.plot_components(forecast)
    fig_param = m.plot_parameters()
    
    plt.show()
    
    return

def alg_4():
    from neuralprophet import NeuralProphet
    from sklearn.metrics import mean_absolute_error
    import time

    st_time = time.time()

    # データセットの分割
    data_size = len(df_true)
    test_length = 10

    train = df_true[:-10]
    test = df_true[-10:]

    # 機械学習モデルのインスタンス作成
    m = NeuralProphet(seasonality_mode='multiplicative')

    # validationの設定(3:1)&モデルのトレーニング
    model = NeuralProphet(seasonality_mode='multiplicative')
    df1_nprophet_model_result = model.fit(train, freq="D")

    # 作ったモデルで予測
    future = model.make_future_dataframe(train, periods=test_length, n_historic_predictions=len(train))
    df_pred = model.predict(future)

    pred_plot = model.plot(df_pred)
    df1_pmpc = model.plot_components(df_pred)
    
    test['NeuralProphet Predict'] = df_pred.iloc[-test_length:].loc[:, 'yhat1']

    print('MAE(Prophet):')
    print(mean_absolute_error(test['y'], test['Prophet Predict']))
    print('MAE(NeuralProphet):')
    print(mean_absolute_error(test['y'], test['NeuralProphet Predict']))
    print('----------------------------')
    print('MAPE(Prophet):')
    print(np.mean(abs(test['y'] - test['Prophet Predict']) / test['y']) * 100)
    print('MAPE(NeuralProphet):')
    print(np.median(abs(test['y'] - test['NeuralProphet Predict']) / test['y']) * 100)
    test.plot(title='Forecast evaluation', ylim=[0, 15])

    # plt.show()

    return


def alg_5():
    from neuralprophet import NeuralProphet
    from sklearn.metrics import mean_absolute_error
    
    # 学習データとテストデータの分割
    df2 = df_true.copy()
    
    test_length = 24
    df2_train = df2.iloc[:-test_length]
    df2_test = df2.iloc[-test_length:]
    
    # 機械学習モデルのインスタンス作成
    params = {
        "growth": "linear",
        "changepoints": None,
        # "n_changepoints": 10,
        "changepoints_range": 0.9,
        "trend_reg": 0,
        "trend_reg_threshold": False,
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "daily_seasonality": False,
        "seasonality_mode": "multiplicative",
        "seasonality_reg": 0,
        "n_forecasts": 1,
        "n_lags": 0,
        "num_hidden_layers": 0,
        "d_hidden": None,
        # "ar_sparsity": None,
        "learning_rate": None,
        "epochs": 10,
        "batch_size": None,
        "loss_func": "Huber",
        # "train_speed": None,
        "normalize": "auto",
        "impute_missing": True,
    }
    
    # NeuralProphet 予測モデル構築
    df2_nprophet_model = NeuralProphet(**params)
    df2_nprophet_model_result = df2_nprophet_model.fit(df2_train, freq="D")
    
    # NeuralProphet 予測モデルの精度検証用データの生成
    df2_future = df2_nprophet_model.make_future_dataframe(df2_train, periods = test_length, n_historic_predictions=len(df2_train))
    df2_pred = df2_nprophet_model.predict(df2_future)

    # # NeuralProphet 予測モデルの予測結果（学習データ期間＋テストデータ期間）
    df2_pred_plot = df2_nprophet_model.plot(df2_pred)         #予想値（点は学習データの実測値）
    df2_pmpc = df2_nprophet_model.plot_components(df2_pred)   #モデルの要素分解
    # fig_param = df2_nprophet_model.plot_parameters(df2_pred)
    
    # テストデータに予測値を結合
    df2_test['new Predict'] = df2_pred.iloc[-test_length:].loc[:, 'yhat1']
    df2_test['current Predict'] = df_old.iloc[-test_length:].loc[:, "y"]
    df2_test.set_index('ds', inplace=True)
    
    # NeuralProphet 予測モデルの精度検証（テストデータ期間）
    # print(mean_absolute_error(df2_test['y'], df2_test['new Predict'])) 
    
    print(df2_test)
    df2_test.plot(title='Forecast evaluation')
    plt.show()
    
    
    return


def alg_6():
    df_compare = pd.read_excel('true_data2.xlsx')
    df_compare['ds'] = pd.to_datetime(df_compare['ds'])
    
    diff_list = [0]*len(df_compare)
    minas_index_list = []
    surplus_index_list = []
    
    for i in range(len(df_compare)):
        # diff_list[i] = abs(df_compare.iloc[i, 1] - df_compare.iloc[i, 2])
        diff_list[i] = df_compare.iloc[i, 1] - df_compare.iloc[i, 2]
        if diff_list[i] > 700:
            surplus_index_list.append(i)
        elif diff_list[i] < -500:
            minas_index_list.append(i)
            
    print(len(surplus_index_list), len(minas_index_list))
    
    pd.set_option('display.max_rows', None)
    print(df_compare.iloc[surplus_index_list, :])
    print(df_compare.iloc[minas_index_list, :])
    
    plt.plot(diff_list)
    plt.show()
    
    return


def now_alg(YYYY, MM, DD, HH):
    # data読み込み
    df = pd.read_excel('true_data.xlsx')
    df['ds'] = pd.to_datetime(df['ds'])
    
    ## 基準時DATETIMEを基に、対象のdfを抽出 (dfをどんどん更新) 
    # 年, 月
    if MM == 2:
        df = df[(df['ds'] >= datetime.datetime(YYYY-1,MM, 1)) & (df['ds'] <= datetime.datetime(YYYY-1,MM,28))]
    elif MM == 4 or MM == 6 or MM == 9 or MM == 11:
        df = df[(df['ds'] >= datetime.datetime(YYYY-1,MM, 1)) & (df['ds'] <= datetime.datetime(YYYY-1,MM,30))]
    else:
        df = df[(df['ds'] >= datetime.datetime(YYYY-1,MM, 1)) & (df['ds'] <= datetime.datetime(YYYY-1,MM,31))]
    df.reset_index(inplace=True)
    
    # 日 (休日・平日)
    Date = datetime.datetime(YYYY,MM, DD, HH)
    delete_index = []
    
    if Date.weekday() >= 5 or jpholiday.is_holiday(Date):
        for i in range(len(df)):
            if df.loc[i, "ds"].weekday() < 5 and jpholiday.is_holiday(df.loc[i, "ds"]) == False:
                delete_index.append(i)
    else:
        for i in range(len(df)):
            if df.loc[i, "ds"].weekday() >= 5 or jpholiday.is_holiday(df.loc[i, "ds"]):
                delete_index.append(i)
    
    df.drop(delete_index, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # 時刻絞り
    delete_index = []
    for i in range(len(df)):
        if df.loc[i, "ds"].hour != HH:
            delete_index.append(i)
    
    df.drop(delete_index, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # print(df)
    
    ## yの平均を出す
    ave = df["y"].median()
    # print(ave)
    
    return ave


def check_old_error(YYYY, MM, DD, HH):
    df_compare = pd.read_excel('true_data2.xlsx')
    df_compare['ds'] = pd.to_datetime(df_compare['ds'])
    df = df_compare.copy()
     
    ## 基準時DATETIMEを基に、対象のdfを抽出 (dfをどんどん更新) 
    # 年, 月
    delete_index = []
    for i in range(len(df)):
        if (df.loc[i, 'ds'].month != MM) or (df.loc[i, 'ds'].day != DD):
            delete_index.append(i)
        
    df.drop(delete_index, inplace=True)
    df.reset_index(inplace=True, drop=True)
    # print(df)
    
    ## 日 (休日・平日)
    Date = datetime.datetime(YYYY,MM, DD, HH)
    delete_index = []
    
    if Date.weekday() >= 5 or jpholiday.is_holiday(Date):
        for i in range(len(df)):
            if df.loc[i, "ds"].weekday() < 5 and jpholiday.is_holiday(df.loc[i, "ds"]) == False:
                delete_index.append(i)
    else:
        for i in range(len(df)):
            if df.loc[i, "ds"].weekday() >= 5 or jpholiday.is_holiday(df.loc[i, "ds"]):
                delete_index.append(i)
    
    df.drop(delete_index, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # # 時刻絞り
    delete_index = []
    for i in range(len(df)):
        if df.loc[i, "ds"].hour != HH:
            delete_index.append(i)
    df.drop(delete_index, inplace=True)
    df.reset_index(inplace=True, drop=True)
    # print(df)
    
    # plt.plot(df['diff'])
    # plt.show()
    
    diff_list = [0]*len(df)
    for i in range(len(df)):
        diff_list[i] = df.iloc[i, 1] - df.iloc[i, 2]
    df['diff'] = diff_list # 予測値-正解値
    
    ave_error = df["diff"].median()
    
    # for i in range(len(df)):
    #     df.loc[i, 'new_y'] = df.loc[i, 'y'] + ave_error
    #     df.loc[i, 'old_error_rate'] = abs(100-100*(df.loc[i, 'old_y'] / df.loc[i, 'y']))
    #     df.loc[i, 'new_error_rate'] = abs(100-100*(df.loc[i, 'new_y'] / df.loc[i, 'y']))
        
    return ave_error 


def get_old_error_rate(df):
        
    print(df["old_error_rate"])
    print("median", df["old_error_rate"].median())
    print("median", df["new_error_rate"].median())
    
    return


def get_error_rate(true_value, pred_value):
    
    # error_rate = abs(100-100*(pred_value / true_value))
    # error_rate = round(error_rate, 2)
    
    # ペナルティを考慮した誤差率の計算
    # penalty = 2
    # error_rate = 100*(pred_value / true_value)
    # if error_rate < 100:
    #     error_rate = penalty*(100 - error_rate)
    # else:
    #     error_rate = error_rate - 100
    
    # 金額
    penalty = 2
    error_cost = pred_value - true_value
    if error_cost < 0:
        error_cost = penalty*error_cost*(-1)
    
    return error_cost


def get_future_error_rate(new_value, old_value, YYYY, MM, DD, HH):
    df = pd.read_excel('true_data2.xlsx')
    df['ds'] = pd.to_datetime(df['ds'])
    
    # 当日の日付・時刻のデータフレームを抽出
    delete_index = []
    for i in range(len(df)):
        if (df.loc[i, "ds"].year != YYYY) or (df.loc[i, 'ds'].month != MM) or (df.loc[i, 'ds'].day != DD) or (df.loc[i, "ds"].hour != HH):
            delete_index.append(i)
            
    df.drop(delete_index, inplace=True)
    df.reset_index(inplace=True, drop=True)
    
    if len(df) == 0:
        print("data is NULL")
        return 0, 0, 0
    
    true_value = df.loc[0, 'y']
    current_value = df.loc[0, 'old_y']
    
    new_error_rate = (round(get_error_rate(true_value, new_value)))*1000
    old_error_rate = (round(get_error_rate(true_value, old_value)))*1000
    cur_error_rate = (round(get_error_rate(true_value, current_value)))*1000
    
    print(f"new: {new_error_rate} yen")
    print(f"old: {old_error_rate} yen")
    print(f"cur: {cur_error_rate} yen")
    
    return new_error_rate, old_error_rate, cur_error_rate


def main():
    new_error_costs = []
    old_error_costs = []
    cur_error_costs = []
    
    for day in range(1, 32):
        for hour in range(24):
            # 予測日時
            YYYY = 2020
            MM = 11
            DD = day
            HH = hour
            print(f"{YYYY}-{MM}-{DD}-{HH}")
            
            ave = now_alg(YYYY, MM, DD, HH)
            error_ave = check_old_error(YYYY, MM, DD, HH) #予測値-正解値
            
            # 新しい予測値の生成
            old_prediced_value = ave
            if error_ave > 0:
                new_prediced_value = ave - error_ave
            else:
                new_prediced_value = ave - 2*error_ave
            
            new, old, cur = get_future_error_rate(new_prediced_value, old_prediced_value, YYYY, MM, DD, HH)
            
            new_error_costs.append(new)
            old_error_costs.append(old)
            cur_error_costs.append(cur)
            
            # print(f" (new: {round(new_prediced_value, 2)}),\n (old: {round(ave, 2)}),\n (diff: {round(error_ave, 2)})")

    # 成績確認 
    print(round(np.mean(new_error_costs)), "yen")
    print(round(np.mean(old_error_costs)), "yen")
    print(round(np.mean(cur_error_costs)), "yen")
    
    return


if __name__ == "__main__":
    # alg_3()
    # alg_4()
    # alg_5()
    
    # alg_6()
    
    main()
    
    