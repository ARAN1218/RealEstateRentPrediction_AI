#必要なライブラリをインポート
from bs4 import BeautifulSoup
import requests
from time import sleep
import json
import pandas as pd
from tqdm import tqdm_notebook as tqdm


#スクレイピングに必要なパラメータを入力
start = 1  #初めのページ数
end = 1000  #終わりのページ数(SUUMOのサイトを見て、何ページまでデータがあるかを確認する)
place = '相模原'  #(辞書urlsに入っている内の)読み込む地域

#後でformatでページ数を代入するので、urlの内「pn=」の部分は「pn={}」としておく
urls = {
    '相模原':"https://suumo.jp/jj/common/ichiran/JJ901FC004/?ar=030&ta=14&sc=14151&sc=14152&sc=14153&kwd=&cb=0.0&ct=9999999&kb=0&kt=9999999&km=1&xb=0&xt=9999999&et=9999999&cn=9999999&newflg=0&pn={}",
    '横浜':"https://suumo.jp/jj/common/ichiran/JJ901FC004/?initFlg=1&seniFlg=1&pc=30&ar=030&ta=14&sa=01&newflg=0&km=1&bs=040&pn={}",
    '渋谷':"https://suumo.jp/jj/common/ichiran/JJ901FC004/?ar=030&ta=13&sc=13113&kwd=&cb=0.0&ct=9999999&kb=0&kt=9999999&km=1&xb=0&xt=9999999&et=9999999&cn=9999999&newflg=0&pn={}",
    '新宿':"https://suumo.jp/jj/common/ichiran/JJ901FC004/?initFlg=1&seniFlg=1&pc=30&ar=030&ta=13&scTmp=13104&kb=0&xb=0&newflg=0&km=1&sc=13104&bs=040&pn={}",
    '港':"https://suumo.jp/jj/common/ichiran/JJ901FC004/?ar=030&ta=13&sc=13103&kwd=&cb=0.0&ct=9999999&kb=0&kt=9999999&km=1&xb=0&xt=9999999&et=9999999&cn=9999999&newflg=0&pn={}"
}


#不動産データスクレイピングのメインプログラム
#Real_Estate_Data_Scraping
def REDS(start,end,place,pre_results=[]):
    #初期値として、pre_resultsを継承する。
    d_list = pre_results
    url = urls[place]
    
    #pre_resultsに値があった場合(再開する場合)、読み込んだページ数をprogressに記録する。
    if len(d_list) > 0:
        progress = list(pd.DataFrame(d_list)['ページ'].unique())
    else:
        progress = []
    
    #tqdmをfor文のrangeに適応することで、スクレイピングの進捗を確認しやすくなる。
    for i in tqdm(range(start,end+1)):
        
        #progressにあるページ番号(既に読み込んだページ番号)は飛ばす。
        if i in progress:
            continue
        
        #途中でエラーが発生してもそれまでの結果が保存できるようにする。
        try:
            
            #ページを遷移させる。
            target_url = url.format(i)

            #Requestsを用いてtarget_urlにアクセスする。
            r = requests.get(target_url)

            #サーバー負荷軽減の為、ループ毎に1秒間隔を空ける。
            sleep(1)

            #取得したHTMLをBeautifulSoupで解析する。
            soup = BeautifulSoup(r.text)

            #BeautifulSoupで解析したHTMLの内、欲しい不動産情報が乗っている部分を取得
            contents = soup.find_all('div',class_='cassettebox js-normalLink js-cassetLink')

            #1ページ当たり30件の不動産データが表示されていれば、リストcontentsは30個の要素を持つはずなので、それらを一つずつ取り出してデータを取得する。
            for content in contents:

                #再開機能の為、取得元のページ数を保存しておく。
                pages = i

                #上の行のデータを取得
                rows = content.find_all('table',class_='listtable')
                address = rows[0].find_all('div',class_='infodatabox-box-txt')[0].text
                station = rows[0].find_all('div',class_='infodatabox-box-txt')[1].text
                access = rows[0].find_all('div',class_='infodatabox-box-txt')[2].text

                #下の行のデータを取得
                r_fees = rows[1].find_all('dd',class_='infodatabox-details-txt')[0].text[:-2]
                mc_fees = rows[1].find_all('dd',class_='infodatabox-details-txt')[1].text[:-1]
                k_fees = rows[1].find_all('dd',class_='infodatabox-details-txt')[2].text.split('/')[0]
                s_fees = rows[1].find_all('dd',class_='infodatabox-details-txt')[2].text.split('/')[1][:-2]
                area = rows[1].find_all('dd',class_='infodatabox-details-txt')[3].text[:-2]
                layout = rows[1].find_all('dd',class_='infodatabox-details-txt')[4].text
                age = rows[1].find_all('div',class_='infodatabox-box-txt')[2].text

                #取得した各種データを辞書dに格納する。
                d = {
                'ページ':pages,
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

                #辞書dのデータをリストd_listに格納する。
                d_list.append(d)

                #重複したデータの削除は、後でまとめてする。
                #d_list = list(map(json.loads,set(map(json.dumps,d_list))))

            #進捗を報告させる。
            print("d_list's progress:",i,"page　　",len(d_list))
            print(target_url)
            
        #リストにある存在しないページにアクセスした場合、そのページの読み込みをスキップする。
        except IndexError:
            continue
            
        #スクレイピングが中断されても、その時点までに読み込んだデータを出力できるようにする。
        except:
            break
    
    #スクレイピングが終わった事を通知
    print('Scraping Completed!')
    return d_list


#中断した場合に進捗を保存しておくリスト
reds_pre_results = []


#中断してもすぐ再開できるように、スクレイピングの結果はreds_testとreds_pre_resultsに共有しておく。
reds_test = REDS(start,end,place,reds_pre_results)
reds_pre_results = reds_test
reds_pre_results


#取得したデータはpickleデータとして保存しておく。
reds_df = pd.DataFrame(reds_test)
reds_df.to_pickle('SUUMO_Sagamihara_Bigdata.pickle')
