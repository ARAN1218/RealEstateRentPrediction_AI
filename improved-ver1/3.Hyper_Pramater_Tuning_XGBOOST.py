#必要なライブラリをインポート
import xgboost as xgb


#hyperoptで使用するためのXGBOOSTモデルのクラス
class Model:

    #初期設定メソッド
    def __init__(self, params=None):
        self.model = None
        if params is None:
            self.params = {}
        else:
            self.params = params

    #学習メソッド
    def fit(self, tr_x, tr_y, va_x, va_y):
        params = {
            #'booster': 'gbtree',
            'objective': 'reg:squarederror',
            'eta': 0.1,
            'gamma': 0.0,
            'alpha': 0.0,
            'lambda': 1.0,
            'min_child_weight': 1,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 71,
        }
        params.update(self.params)
        num_round = 20
        dtrain = xgb.DMatrix(tr_x, label=tr_y)
        dvalid = xgb.DMatrix(va_x, label=va_y)
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        self.model = xgb.train(params, dtrain, num_round, early_stopping_rounds=10, evals=watchlist)

    #予測メソッド
    def predict(self, x):
        data = xgb.DMatrix(x)
        pred = self.model.predict(data)
        return pred
      
      
      
      
#必要なライブラリをインポート
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import mean_squared_error


#hyperoptを使ったパラメータ探索
def score(params):
    # パラメータを与えたときに最小化する評価指標を指定する
    # 具体的には、モデルにパラメータを指定して学習・予測させた場合のスコアを返すようにする

    # max_depthの型を整数型に修正する
    params['max_depth'] = int(params['max_depth'])

    # Modelクラスは、fitで学習し、predictで予測値の確率を出力する
    model = Model(params)
    model.fit(tr_x, tr_y, va_x, va_y)
    va_pred = model.predict(va_x)
    score = np.sqrt(mean_squared_error(va_y, va_pred)) #rmseを最小化するようにパラメータをチューニング
    print(f'params: {params}, rmse: {score:.4f}')

    # 情報を記録しておく
    history.append((params, score))

    return {'loss': score, 'status': STATUS_OK}


#データ毎に最適なハイパーパラメータチューニングをhyperoptで行い、最も良かったパラメータを返す。
#Hyper_Pramater_Tuning_XGBOOST
def HPT_XGB(df_l):
    train_x = df_l.drop(['賃料'], axis=1)
    train_y = df_l['賃料']

    # 学習データを学習データとバリデーションデータに分ける
    kf = KFold(n_splits=4, shuffle=True, random_state=71)
    tr_idx, va_idx = list(kf.split(train_x))[0]
    global tr_x,va_x,tr_y,va_y #score関数で利用するため、グローバル変数として扱う。
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

    # hp.choiceでは、複数の選択肢から選ぶ
    # hp.uniformでは、下限・上限を指定した一様分布から抽出する。引数は下限・上限
    # hp.quniformでは、下限・上限を指定した一様分布のうち一定の間隔ごとの点から抽出する。引数は下限・上限・間隔
    # hp.loguniformでは、下限・上限を指定した対数が一様分布に従う分布から抽出する。引数は下限・上限の対数をとった値

    # 探索するパラメータの空間を指定
    param_space = {
        'max_depth': hp.quniform('max_depth', 3, 9, 1),
        'min_child_weight': hp.loguniform('min_child_weight', np.log(0.1), np.log(10)),
        'subsample': hp.quniform('subsample', 0.6, 0.95, 0.05),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.6, 0.95, 0.05),
        'gamma': hp.loguniform('gamma', np.log(1e-8), np.log(1.0)),
        'alpha' : hp.loguniform('alpha', np.log(1e-8), np.log(1.0)),
        'lambda' : hp.loguniform('lambda', np.log(1e-6), np.log(10.0)),
    }

    # hyperoptによるパラメータ探索の実行
    max_evals = 100 #100回探索する。
    trials = Trials()
    global history #score関数で利用するため、グローバル変数として扱う。
    history = []
    fmin(score, param_space, algo=tpe.suggest, trials=trials, max_evals=max_evals)

    # 記録した情報からパラメータとスコアを出力
    #（trialsからも情報が取得できるが、パラメータの取得がやや行いづらい）
    history = sorted(history, key=lambda tpl: tpl[1])
    best = history[0]
    print("\n",f'best params:{best[0]}, score:{best[1]:.4f}')

    return best[0]


#HPT_XGB(学習データ)
best_params = HPT_XGB(df_l)

#HPT_XGB関数のmax_evalsやModelクラスのnum_roundsをいじれば、時間はかかるがより精度の高いチューニングが出来る。
