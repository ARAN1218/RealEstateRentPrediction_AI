#必要なライブラリをインポート
from bs4 import BeautifulSoup
import requests
from time import sleep
import json
import pandas as pd


#スクレイピングに必要なパラメータを入力する
start = 1  #初めのページ数
end = 20  #終わりのページ数
place = '渋谷'  #(辞書urlsに入っている内の)読み込む地域

#後でformatでページ数を代入するので、urlの内「pn=」の部分は「pn={}」としておく
urls = {
    '相模原':"https://suumo.jp/jj/common/ichiran/JJ901FC004/?ar=030&ta=14&sc=14151&sc=14152&sc=14153&kwd=&cb=0.0&ct=9999999&kb=0&kt=9999999&km=1&xb=0&xt=9999999&et=9999999&cn=9999999&newflg=0&pn={}",
    '渋谷':"https://suumo.jp/jj/common/ichiran/JJ901FC004/?ar=030&ta=13&sc=13113&kwd=&cb=0.0&ct=9999999&kb=0&kt=9999999&km=1&xb=0&xt=9999999&et=9999999&cn=9999999&newflg=0&pn={}",
    '新宿':"https://suumo.jp/jj/common/ichiran/JJ901FC004/?ar=030&ta=13&sc=13104&kwd=&cb=0.0&ct=9999999&kb=0&kt=9999999&km=1&xb=0&xt=9999999&et=9999999&cn=9999999&newflg=0&pn={}"
}


#不動産データスクレイピングのメインプログラム
#Real_Estate_Data_Scraping
def REDS(start,end,place):
    #初期値
    d_list = []
    url = urls[place]
    
    #range(start,end+1,2)の内、2の部分を消せば指定した全ページのデータが取得できる
    for i in range(start,end+1,2):
    
        #ページを遷移させる
        target_url = url.format(i)

        #Requestsを用いてtarget_urlにアクセスする
        r = requests.get(target_url)

        #サーバー負荷軽減の為、ループ毎に1秒間隔を空ける
        sleep(1)

        #取得したHTMLをBeautifulSoupで解析する
        soup = BeautifulSoup(r.text)
        
        #BeautifulSoupで解析したHTMLの内、欲しい不動産情報が乗っている部分を取得する
        contents = soup.find_all('div',class_='cassettebox js-normalLink js-cassetLink')
        
        #1ページ当たり30件の不動産データが表示されていれば、リストcontentsは30個の要素を持つはずなので、それらを一つずつ取り出してデータを取得する
        for content in contents:
            
            #上の行のデータを取得する
            rows = content.find_all('table',class_='listtable')
            address = rows[0].find_all('div',class_='infodatabox-box-txt')[0].text
            station = rows[0].find_all('div',class_='infodatabox-box-txt')[1].text
            access = rows[0].find_all('div',class_='infodatabox-box-txt')[2].text

            #下の行のデータを取得する
            r_fees = rows[1].find_all('dd',class_='infodatabox-details-txt')[0].text[:-2]
            mc_fees = rows[1].find_all('dd',class_='infodatabox-details-txt')[1].text[:-1]
            k_fees = rows[1].find_all('dd',class_='infodatabox-details-txt')[2].text.split('/')[0]
            s_fees = rows[1].find_all('dd',class_='infodatabox-details-txt')[2].text.split('/')[1][:-2]
            area = rows[1].find_all('dd',class_='infodatabox-details-txt')[3].text[:-2]
            layout = rows[1].find_all('dd',class_='infodatabox-details-txt')[4].text
            age = rows[1].find_all('div',class_='infodatabox-box-txt')[2].text

            #取得した各種データを辞書dに格納する
            d = {
            '住所':address,
            '路線':station,
            '交通':access,
            '賃料':r_fees,
            '管理共益費':mc_fees,
            '礼金':k_fees,
            '敷金':s_fees,
            '専有面積':area,
            '間取り':layout,
            '築年数':age
            }

            #辞書dのデータをリストd_listに格納する
            d_list.append(d)
            
            #重複したデータを削除する
            d_list = list(map(json.loads,set(map(json.dumps,d_list))))

        #進捗を報告させる
        print("d_list's progress:",i,"page　　",len(d_list))
        print(target_url)
    
    #スクレイピングが終わった事を通知する
    print('Scraping Completed!')
    return d_list


#学習データとして、目的のページ数までの奇数ページのデータを取得する。
#テストデータとして、目的のページ数までの偶数ページのデータを取得する。
#返り値でリストが来るため、リストとして変数に代入させる。
#REDS(最初のページ数, 最後のページ数, 地域)
d_list = REDS(start,end,place)
d_t_list = REDS(start+1,end+1,place)

#スクレイピングした不動産データをデータフレームに変換し、上位10件を表示させる。
df = pd.DataFrame(d_list)
df_test = pd.DataFrame(d_t_list)
display(df.head(10))
display(df_test.head(10))
