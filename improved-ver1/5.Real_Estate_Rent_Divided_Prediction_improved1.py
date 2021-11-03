#必要なライブラリをインポート
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import warnings


#種々の警告を非表示にする。
warnings.simplefilter('ignore')


#pandasのデータフレームの列表示制限を解除
pd.set_option('display.max_rows',None)


#既に学習データとREDP関数を用いてAIモデルが作成されているものとする
#model = RERL(df_l)


#RERLで作ったAIを用いて、テストデータの不動産の賃料を予測する関数
#Real_Estate_Rent_Divided_Prediction
def RERDP(df_t):
    #説明変数と目的変数を分割
    df_fact = df_t['賃料']
    df_input = df_t.drop(['賃料'],axis=1)
    
    #予測値の生成
    df_input = xgb.DMatrix(df_input)
    pred = model.predict(df_input)
    
    #実値と予測値の表示
    df_pred = pd.DataFrame(pred,columns=['予測値'])
    df_final = pd.concat([df_fact,df_pred],axis=1)
    display(df_final)
    
    #MAPEとSMAPEの計算のため、df_factとdf_predをndarrayに変換する(MAE等の計算に影響しない)
    df_fact = df_fact.to_numpy()
    df_pred = df_pred.to_numpy()
    
    #モデルの性能を表示
    print('MAE:',mean_absolute_error(df_fact,df_pred)) #予測値と実測値の平均的なズレ（誤差）の大きさ
    print('MSE:',mean_squared_error(df_fact,df_pred)) #予測値と実測値のズレの大きさ(大きな誤差を重視)
    print('RMSE:',np.sqrt(mean_squared_error(df_fact,df_pred))) #MSEの平方根
    print('R^2:',r2_score(df_fact, df_pred)) #回帰モデルの当てはまりの良さ(どれだけ値が近いか)
    print('MAPE:',np.mean(np.abs((df_fact - df_pred) / df_fact)) * 100) #実測値の大きさに対する予測値の平均的なズレ（誤差）の「割合」(実測値が0を取るケースでは使用できない　)
    print('SMAPE:',100 * (1/len(df_fact)) * np.sum(2 * np.abs(df_pred - df_fact) / (np.abs(df_pred) + np.abs(df_fact)))) #実測値の大きさに対する予測値の平均的なズレ（誤差）の「割合」(実測値が0を取るケースでも使用できる。)
    

#RERDP(テストデータ)
RERDP(df_t)
