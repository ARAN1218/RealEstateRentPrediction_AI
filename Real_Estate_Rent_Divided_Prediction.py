#必要なライブラリをインポート
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import xgboost as xgb

#pandasのデータフレームの列表示制限を解除
pd.set_option('display.max_rows',None)


#既に学習データとREDP関数を用いてAIが作成されているものとする
#model = RERL(df_l)


#RERLで作ったAIを用いて、テストデータの不動産の賃料を予測する関数
#Real_Estate_Rent_Divided_Prediction
def RERDP(df):
    #学習データと教師データの分割
    df_fact = df['賃料']
    df_input = df.drop(['賃料'],axis=1)
    
    #予測値の生成
    df_input = xgb.DMatrix(df_input)
    pred = model.predict(df_input)
    
    #実値と予測値の表示
    df_pred = pd.DataFrame(pred,columns=['予測値'])
    df_final = pd.concat([df_fact,df_pred],axis=1)
    display(df_final)
    
    #モデルの性能を表示
    print('MAE:',mean_absolute_error(df_fact,df_pred))
    print('MSE:',mean_squared_error(df_fact,df_pred))
    print('RMSE:',np.sqrt(mean_squared_error(df_fact,df_pred)))


#RERDP(テストデータ)
RERDP(df_t)
