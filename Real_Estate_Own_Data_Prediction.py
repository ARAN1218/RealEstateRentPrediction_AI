#必要なライブラリをインポート
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder


#既に学習データとREDP関数を用いてAIが作成されているものとする
#model = RERL(df_l)


#住所と路線と間取りはラベルエンコーディングの都合により学習データ/テストデータにあったものしか使えない為、予め確保しておいた使える要素を表示させる
#上記の3つの要素はこの中から選んでもらう
#このプログラムは別のセルで起動させると見やすい
print(adressC,'\n','\n',stationC,'\n','\n',layoutC)


#(学習した範囲内の)任意のデータを入力して予測できる関数
#Real_Estate_Own_Data_Prediction
def REODP(address,station,access,mc_fees,k_fees,s_fees,area,layout,age):
    #入力したデータを辞書d_tryに格納する
    d_try = {
        '住所':address,
        '路線':station,
        '交通':access,
        '管理共益費':mc_fees,
        '礼金':k_fees,
        '敷金':s_fees,
        '専有面積':area,
        '間取り':layout,
        '築年数':age
    }
    
    #辞書d_tryをデータフレームdf_tryに変換する
    df_try = pd.DataFrame(d_try,index=['own'])
    
    #入力情報の確認用
    display(df_try)
    
    #ラベルエンコーディングを行い、文字列を数値化する
    df_try.住所 = LE1.transform(df_try.住所)
    df_try.路線 = LE2.transform(df_try.路線)
    df_try.間取り = LE3.transform(df_try.間取り)
    
    #データ型をfloat64で統一する
    df_try = df_try.astype('float64')
    
    #予測結果を表示する
    df_try = xgb.DMatrix(df_try)
    return print('予想賃料:',float(model.predict(df_try)),'万円')


#REODP(住所, 路線, 交通, 管理共益費, 礼金, 敷金, 専有面積, 間取り, 築年数)
#データ型に気をつける
#住所と路線と間取りはラベルエンコーディングの都合により学習データ/テストデータにあったものしか使えない為、上で表示させた要素から選ぶこと
REODP(address=''
      ,station=''
      ,access=0
      ,mc_fees=0
      ,k_fees=0
      ,s_fees=0
      ,area=0
      ,layout=''
      ,age=0
     )
