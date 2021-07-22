#必要なライブラリのインポート
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import xgboost as xgb
import japanize_matplotlib #matplotlibでグラフを描写する際、日本語が文字化けするのを防ぐ。
import shap
import warnings


#種々の警告を非表示にする。
warnings.simplefilter('ignore')


#pandasのデータフレームの列表示制限を解除
pd.set_option('display.max_rows',None)


#既にREDPD関数でデータ前処理が終わっており、HPT_XGB関数でパラメータを取得しているとする
#df_l,df_t = REDPD(df)
#best_params = HPT_XGB(df_l)


#不動産賃料予測AIモデルの学習プログラム
#HPT_XGBで取得したパラメータを利用する
#Real_Estate_Rent_Learning
def RERL(df_l,best_params):
    #学習データと教師データに分ける
    target = df_l['賃料']
    train = df_l.drop(['賃料'],axis=1)
    
    #クロスバリデーション
    kf = KFold(n_splits=4,shuffle=True,random_state=71)
    tr_idx,va_idx = list(kf.split(train))[0]
    tr_x,va_x = train.iloc[tr_idx],train.iloc[va_idx]
    tr_y,va_y = target.iloc[tr_idx],target.iloc[va_idx]
    
    #データ型をXGBOOST用に適合させる
    dtrain = xgb.DMatrix(tr_x,label=tr_y)
    dvalid = xgb.DMatrix(va_x,label=va_y)
    
    #XGBOOSTのモデルを作成
    num_round = 1000
    early_stopping_rounds=20
    watchlist = [(dtrain,'train'),(dvalid,'eval')]
    model = xgb.train(best_params,dtrain,num_round,early_stopping_rounds=early_stopping_rounds,evals=watchlist)
    
    
    #テストデータとその予測結果を表示
    va_pred = model.predict(dvalid)
    df_true = pd.DataFrame(list(va_y),columns=['賃料'])
    df_pred = pd.DataFrame(va_pred,columns=['予測値'])
    display(pd.concat([df_true,df_pred],axis=1))
    
    #モデルの性能を表示
    print('MAE:',mean_absolute_error(va_y,va_pred)) #予測値と実測値の平均的なズレ（誤差）の大きさ
    print('MSE:',mean_squared_error(va_y,va_pred)) #予測値と実測値のズレの大きさ(大きな誤差を重視)
    print('RMSE:',np.sqrt(mean_squared_error(va_y,va_pred))) #MSEの平方根
    
    #変数の予測への寄与率をグラフにして表示
    shap.initjs()
    explainer = shap.TreeExplainer(model = model)
    shap_values = explainer.shap_values(X = tr_x)
    shap.summary_plot(shap_values, tr_x)
    shap.summary_plot(shap_values, tr_x, plot_type = "bar")
    #上のグラフは、横軸がその変数自体の大きさ、色が目的変数への影響の違い(赤い程目的変数を大きくする)を表す
    
    return model

#RERL(学習データ)
model = RERL(df_l,best_params)
