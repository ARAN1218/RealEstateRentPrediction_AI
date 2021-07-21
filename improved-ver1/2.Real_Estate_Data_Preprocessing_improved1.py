#----------------------------------
#生データの大まかな処理
#----------------------------------
#必要なライブラリをインポート
import pandas as pd
from sklearn.model_selection import train_test_split


#REDS関数でスクレイピングして保存したpickleファイルを読み込む。
reds_df = pd.read_pickle('SUUMO_Sagamihara_Bigdata.pickle')


#重複処理前のデータ数を確認してみる。
print("処理前データ数：",len(reds_df))


#「ページ」カラムは要らないので削除する。
reds_df.drop('ページ',axis=1,inplace=True)


#重複しているデータを処理する。
reds_df = reds_df.drop_duplicates()


#重複処理後のデータ数を確認してみる。
print("処理後データ数：",len(reds_df))


#教師データ：テストデータ＝3:1になるように、データをシャッフルして分割する。
df,df_t = train_test_split(reds_df,test_size=0.25,random_state=71,shuffle=True)
print(sorted(df['賃料'].unique().astype(float)),"\n")
print(sorted(df_t['賃料'].unique().astype(float)))


#データ分割後のそれぞれのデータ数を確認してみる。
print("処理後学習データ数：",len(df))
print("処理後テストデータ数：",len(df_t))
#処理後学習データ数 + 処理後テストデータ数　＝　処理後データ数　となっていたら正常に動作している。


#----------------------------------
#データ前処理
#----------------------------------
#必要なライブラリをインポート
import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder


#学習データ/テストデータを同時に前処理する関数
#Real_Estate_Data_Preprocessing
def REDP(d_df,d_t_df):
    #バラバラになっているインデックスを振り直して整理する。
    df_l = d_df.reset_index(drop=True)
    df_t = d_t_df.reset_index(drop=True)
    
    #df内の余計な文字(\n等)を消去する
    for df in [df_l,df_t]:
        df['礼金'] = df['礼金'].str.replace(r'\(.+\)','')
        for element in ['住所','交通','専有面積','間取り','築年数','礼金','敷金']:
            df[element] = df[element].str.translate(str.maketrans({'-':'','\r':'','\n':'','\t':''
                                                                   ,'バ':'','ス':'','分':'','徒':''
                                                                   ,'歩':'','停':'','―':'','築':''
                                                                   ,'年':'','新':'','万':'','(':''
                                                                   ,'円':'','車':'','以':'','上':''}))
    
    #オリジナル予測の為にラベルエンコーダーをグローバル変数として用意する
    global LE1,LE2,LE3
    LE1,LE2,LE3 = LabelEncoder(),LabelEncoder(),LabelEncoder()
    
    #ラベルエンコーダーの学習を行い、列毎に変換の数字を合わせる
    #注意:余計な文字を取り除いた段階でテストデータも一緒にラベルエンコードしないと、テストデータにのみ存在する住所等を変換できなくなってしまう為、学習データと縦に連結してfitさせる
    LE_df = pd.concat([df_l,df_t])
    LE1.fit(LE_df['住所'])
    LE2.fit(LE_df['路線'])
    LE3.fit(LE_df['間取り'])
    
    #ラベルエンコーディングを行い、文字列を数値化する
    for df in [df_l,df_t]:
        df['住所'] = LE1.transform(df['住所'])
        df['路線'] = LE2.transform(df['路線'])
        df['間取り'] = LE3.transform(df['間取り'])
    
        #欠損値を0に統一する
        df.replace('',0,inplace=True)
    
    #データ型をfloat64で統一する
    #astypeの使用上、for文に組み込めなかった
    df_l = df_l.astype('float64')
    df_t = df_t.astype('float64')
    
    return df_l,df_t


#REDP(前処理したいリスト学習データ,前処理したいリストテストデータ)
df_l,df_t = REDP(df,df_t)


#確認用
display(df_l.head(10))
display(df_t.head(10))


#オリジナル予測の為にラベルエンコーダーの学習した要素を変数に格納しておく
adressC,stationC,layoutC = LE1.classes_, LE2.classes_ ,LE3.classes_
print(adressC,'\n','\n',stationC,'\n','\n',layoutC)

#「ページ」カラムを切り取り、重複を外し、データを分けたものを入れる必要がある。
#SettingWithCopyWarningが出ない。


#----------------------------------
#生データから一気にデータ前処理
#----------------------------------
#必要なライブラリをインポート
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings


#SettingWithCopyWarningを無効化する。
warnings.simplefilter('ignore')


#学習データ/テストデータを同時に前処理する関数
#Real_Estate_Data_Preprocessing_Direct
def REDPD(df):
    
    #「ページ」カラムは要らないので削除する。
    df.drop('ページ',axis=1,inplace=True)
    df = df.drop_duplicates()
    
    
    #df内の余計な文字(\n等)を消去する。
    df['礼金'] = df['礼金'].str.replace(r'\(.+\)','')
    for element in ['住所','交通','専有面積','間取り','築年数','礼金','敷金']:
        df[element] = df[element].str.translate(str.maketrans({'-':'','\r':'','\n':'','\t':''
                                                                ,'バ':'','ス':'','分':'','徒':''
                                                                ,'歩':'','停':'','―':'','築':''
                                                                ,'年':'','新':'','万':'','(':''
                                                                ,'円':'','車':'','以':'','上':''}))
    
    #オリジナル予測の為にラベルエンコーダーをグローバル変数として用意する
    global LE1,LE2,LE3
    LE1,LE2,LE3 = LabelEncoder(),LabelEncoder(),LabelEncoder()
    
    #ラベルエンコーダーの学習を行い、列毎に変換の数字を合わせる
    #注意:余計な文字を取り除いた段階でテストデータも一緒にラベルエンコードしないと、テストデータにのみ存在する住所等を変換できなくなってしまう為、学習データと縦に連結してfitさせる
    LE1.fit(df['住所'])
    LE2.fit(df['路線'])
    LE3.fit(df['間取り'])
    
    #ラベルエンコーディングを行い、文字列を数値化する
    df['住所'] = LE1.transform(df['住所'])
    df['路線'] = LE2.transform(df['路線'])
    df['間取り'] = LE3.transform(df['間取り'])
    
    #欠損値を0に統一する
    df.replace('',0,inplace=True)
    
    #データ型をfloat64で統一する
    #astypeの使用上、for文に組み込めなかった
    df = df.astype('float64')
    
    #教師データ：テストデータ＝3:1になるように、データをシャッフルして分割する。
    df_l,df_t = train_test_split(df,test_size=0.25,random_state=71,shuffle=True)
    
    #バラバラになっているインデックスを振り直して整理する。
    df_l = df_l.reset_index(drop=True)
    df_t = df_t.reset_index(drop=True)
    
    return df_l,df_t


#REDPD(前処理したいリスト学習データ,前処理したいリストテストデータ)
df = pd.read_pickle('SUUMO_Sagamihara_Bigdata.pickle')
df_l,df_t = REDPD(df)


#確認用
display(df_l.head(10))
display(df_t.head(10))


#オリジナル予測の為にラベルエンコーダーの学習した要素を変数に格納しておく
adressC,stationC,layoutC = LE1.classes_, LE2.classes_ ,LE3.classes_
print(adressC,'\n','\n',stationC,'\n','\n',layoutC)


#警告を再度表示できるようにする。
warnings.resetwarnings()

#for文が少ない分、こちらの方が処理が早く終わる。
#SettingWithCopyWarningは無効化すれば良い。
