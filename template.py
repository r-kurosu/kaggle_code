from cgi import test
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn import tree
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import tree
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, log_loss



for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


def kesson_table(df): 
    null_val = df.isnull().sum()
    percent = 100 * df.isnull().sum()/len(df)
    kesson_table = pd.concat([null_val, percent], axis=1)
    kesson_table_ren_columns = kesson_table.rename(
    columns = {0 : '欠損数', 1 : '%'})
    
    return kesson_table_ren_columns


def data_preprocessing(train, test):
    train["Age"] = train["Age"].fillna(train["Age"].median())
    train["Embarked"] = train["Embarked"].fillna("S")
    train["Sex"][train["Sex"] == "male"] = 0
    train["Sex"][train["Sex"] == "female"] = 1
    train["Embarked"][train["Embarked"] == "S" ] = 0
    train["Embarked"][train["Embarked"] == "C" ] = 1
    train["Embarked"][train["Embarked"] == "Q"] = 2
    
    test["Age"] = test["Age"].fillna(test["Age"].median())
    test["Sex"][test["Sex"] == "male"] = 0
    test["Sex"][test["Sex"] == "female"] = 1
    test["Embarked"][test["Embarked"] == "S"] = 0
    test["Embarked"][test["Embarked"] == "C"] = 1
    test["Embarked"][test["Embarked"] == "Q"] = 2
    test.Fare[152] = test.Fare.median()
    
    return train, test

def find_best_alfa(target, train_prediction_prob):
    max_score = 0
    alfa_list = list(range(100))
    for a in alfa_list:
        a = a/100
        temp_score = accuracy_score(target, np.where(train_prediction_prob > a, 1, 0))
        if temp_score >= max_score:
            max_score = temp_score
            alfa = a

    return alfa


def xgb_model(train, test):
    features_col = ["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]
    x_train = train[features_col].values
    target = train["Survived"].values
    x_test = test[features_col].values
    
    ht_param = {
                    "max_depth": [i for i in range(4,7)],  # デフォルト6
                    "min_child_weight": [0, 0.5, 1, 1.5],  # デフォルト1
                    "eta": [0.05, 0.1, 0.15, 0.2],  # 0.01~0.2が多いらしい
                    "tree_method": ["exact"],
                    "predictor": ["cpu_predictor"],
                    "lambda": [1],  # 重みに関するL"正則 デフォルト1
                    "alpha": [0],  # 重みに関するL1正則  # デフォルト0
                    "subsample": [0.5, 0.7, 1],  # デフォルト1, 0~1
                    "max_delta_step": [1],
                    'objective': ['binary:logistic'],
                    "eval_metric": ["logloss"],  # 損失関数 l(y, a)
                    "seed": [0]
                }

    # #交差検証+グリッドサーチにより最良パラメータの検索
    # clf = GridSearchCV(xgb.XGBClassifier(), ht_param, cv=5, scoring="accuracy")
    # clf.fit(features, target)
    # print("最良パラメータ: {}".format(clf.best_params_))
    # print("最良交差検証スコア: {:.2f}".format(clf.best_score_))

    # train
    xgb_tree = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        max_depth=5, 
        min_child_weight=0.5, 
        eta=0.1, 
        alpha=0,
        subsample=0.7,
        learning_rate=0.01, 
        n_estimators=50, 
        random_state=3
        )
    xgb_tree.fit(x_train, target)

    train_prediction_prob = xgb_tree.predict(x_train)
    test_prediction_prob = xgb_tree.predict(x_test)
    
    alfa = find_best_alfa(target, train_prediction_prob)
    
    train_prediction = np.where(train_prediction_prob > alfa, 1, 0)
    test_prediction = np.where(test_prediction_prob > alfa, 1, 0)

    return train_prediction, test_prediction


def lgb_model(train, test):
    features_col = ["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]
    x_train = train[features_col].values
    target = train["Survived"].values
    x_test = test[features_col].values

    lgb_train = lgb.Dataset(x_train, label=target)

    ht_params = {"objective": "binary", 
              "seed": 71, 
              "verbose":0, 
              "metrics":"binary_error",
              "force_row_wise":True,
              "num__leaves": 5,
              "reg_alfa":0.1,
              "reg_lambda":20,
             }

    num_round = 100

    model = lgb.train(ht_params, lgb_train, num_boost_round=num_round)

    train_prediction_prob = model.predict(x_train)
    test_prediction_prob = model.predict(x_test)

    alfa = find_best_alfa(target, train_prediction_prob)

    train_prediction = np.where(train_prediction_prob > alfa, 1, 0)
    test_prediction = np.where(test_prediction_prob > alfa, 1, 0)

    return train_prediction, test_prediction


def output_result(train, test, train_prediction, test_prediction):
    y_true = train["Survived"].tolist()
    print("train score: ", accuracy_score(y_true, train_prediction))

    # PassengerIdを取得
    PassengerId = np.array(test["PassengerId"]).astype(int)
    
    # my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む
    my_solution = pd.DataFrame(test_prediction, PassengerId, columns = ["Survived"])
    my_solution.to_csv("submission.csv", index_label = ["PassengerId"])
    
    return


def main():
    train = pd.read_csv("../input/titanic/train.csv")
    test = pd.read_csv("../input/titanic/test.csv")
    
    train, test = data_preprocessing(train, test)
    
    train_prediction, test_prediction = xgb_model(train, test)
    # train_prediction, test_prediction = lgb_model(train, test)
    
    output_result(train, test, train_prediction, test_prediction)
    
    return

if __name__ == "__main__":
    train = pd.read_csv("../input/titanic/train.csv")
    test = pd.read_csv("../input/titanic/test.csv")
    
    main(train, test)