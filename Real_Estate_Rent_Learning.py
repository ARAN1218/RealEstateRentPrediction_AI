#必要なライブラリのインポート
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import xgboost as xgb


#既にREDP関数で学習データ/テストデータの前処理が終わっているとする
#df_l,df_t = REDP(d_list,d_t_list)


#不動産賃料予測AIモデルの学習プログラム
#Real_Estate_Rent_Learning
def RERL(df):
    #学習データと教師データに分ける
    target = df['賃料']
    train = df.drop(['賃料'],axis=1)
    
    #クロスバリデーション
    kf = KFold(n_splits=4,shuffle=True,random_state=71)
    tr_idx,va_idx = list(kf.split(train))[0]
    tr_x,va_x = train.iloc[tr_idx],train.iloc[va_idx]
    tr_y,va_y = target.iloc[tr_idx],target.iloc[va_idx]
    
    #データ型をXGBOOST用に適合させる
    dtrain = xgb.DMatrix(tr_x,label=tr_y)
    dvalid = xgb.DMatrix(va_x,label=va_y)
    
    #パラメータ準備
    params = {'objective':'reg:squarederror','silent':1,'random_state':71}
    num_round = 100

    #XGBOOSTモデルに機械学習
    #100回学習する事になっているが、early_stopping_roundsを設定しているため、10回以内で学習成果が得られない場合は学習を打ち切りにし、過学習を防ぐ
    watchlist = [(dtrain,'train'),(dvalid,'eval')]
    model = xgb.train(params,dtrain,num_round,early_stopping_rounds=10,evals=watchlist)
    va_pred = model.predict(dvalid)
    
    #テストデータとその予測結果を表示
    print(list(va_y))
    print(va_pred)
    
    #モデルの性能を表示
    print('MAE:',mean_absolute_error(va_y,va_pred))
    print('MSE:',mean_squared_error(va_y,va_pred))
    print('RMSE:',np.sqrt(mean_squared_error(va_y,va_pred)))
    
    return model

#RERL(学習データ)
model = RERL(df_l)
