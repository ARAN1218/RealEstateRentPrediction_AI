#必要なライブラリをインポート
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#pandasのデータフレームの列表示制限を解除
pd.set_option('display.max_rows',None)


#既にREDSを用いて変数d_listとd_t_listにスクレイピングしたデータが保存され、それぞれdf,df_testにデータフレーム型に変換されて代入されているとする
#d_list = REDS(start,end,place)
#d_t_list = REDS(start+1,end+1,place)


#オリジナル予測の為にラベルエンコーダーを関数の外で用意する
LE1,LE2,LE3 = LabelEncoder(),LabelEncoder(),LabelEncoder()


#学習データ/テストデータを同時に前処理する関数
#Real_Estate_Data_Preprocessing
def REDP(d_list,d_t_list):
    #データフレームに変換する
    df_l = pd.DataFrame(d_list)
    df_t = pd.DataFrame(d_t_list)
    
    #df内の余計な文字(\n等)を消去する
    for df in [df_l,df_t]:
        df['礼金'] = df['礼金'].str.replace(r'\(.+\)','')
        for element in ['住所','交通','専有面積','間取り','築年数','礼金','敷金']:
            df[element] = df[element].str.translate(str.maketrans({'-':'','\r':'','\n':'','\t':''
                                                                   ,'バ':'','ス':'','分':'','徒':''
                                                                   ,'歩':'','停':'','―':'','築':''
                                                                   ,'年':'','新':'','万':'','(':'','円':''}))
    
    
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
df_l,df_t = REDP(d_list,d_t_list)

#確認用
display(df_l.head(10))
display(df_t.head(10))


#オリジナル予測の為にラベルエンコーダーの学習した要素を変数に格納しておく
adressC,stationC,layoutC = LE1.classes_,LE2.classes_,LE3.classes_
print(adressC,'\n','\n',stationC,'\n','\n',layoutC)
