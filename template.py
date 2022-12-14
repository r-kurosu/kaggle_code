from pyexpat import model
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import tree
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, log_loss
import optuna
from sklearn.model_selection import cross_validate, cross_val_predict, cross_val_score
from sklearn.model_selection import train_test_split


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# optunaの目的関数を設定する
def objective_xgb(trial, x_train, y_train):
    eta =  trial.suggest_loguniform('eta', 1e-8, 1.0)
    gamma = trial.suggest_loguniform('gamma', 1e-8, 1.0)
    max_depth = trial.suggest_int('max_depth', 1, 20)
    min_child_weight = trial.suggest_loguniform('min_child_weight', 1e-8, 1.0)
    max_delta_step = trial.suggest_loguniform('max_delta_step', 1e-8, 1.0)
    subsample = trial.suggest_uniform('subsample', 0.0, 1.0)
    reg_lambda = trial.suggest_uniform('reg_lambda', 0.0, 1000.0)
    reg_alpha = trial.suggest_uniform('reg_alpha', 0.0, 1000.0)


    model =xgb.XGBClassifier(eta = eta, gamma = gamma, max_depth = max_depth,
                           min_child_weight = min_child_weight, max_delta_step = max_delta_step,
                           subsample = subsample,reg_lambda = reg_lambda,reg_alpha = reg_alpha)

    score = cross_val_score(model, x_train, y_train, cv=5, scoring="accuracy")
    accuracy_mean = score.mean()
    print(accuracy_mean)

    return accuracy_mean


def objective_lgb(trial, x_train, y_train):
    from optuna.integration import lightgbm as lgb
    
    train_x, test_x, train_y, test_y = train_test_split(x_train, y_train, test_size=0.25)
    dtrain = lgb.Dataset(train_x, label=train_y)
    dtest = lgb.Dataset(test_x, label=test_y)
    
    param = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt'
        # 'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        # 'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        # 'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        # 'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        # 'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        # 'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        # 'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }
    
    model = lgb.train(param, dtrain, valid_sets=dtest, early_stopping_rounds=100)
    print("Best Params:", model.params)
    
    preds = model.predict(test_x)
    pred_labels = np.rint(preds)
    accuracy = accuracy_score(test_y, pred_labels)
    
    return accuracy


def use_optuna_for_xgb(train):
    features_col = ["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]
    x_train = train[features_col].values
    target = train["Survived"].values
    
    with redirect_stdout(open(os.devnull, 'w')):
        optuna.logging.disable_default_handler()
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective_xgb(trial, x_train, target), n_trials=100)

    # ベストパラメータを出力
    print("params:", study.best_params)
    print("best value: ", study.best_value)
    
    return study.best_params


def use_optuna_for_lgb(train):
    features_col = ["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]
    x_train = train[features_col].values
    target = train["Survived"].values
    
    with redirect_stderr(open(os.devnull, 'w')):
        optuna.logging.disable_default_handler()
        study = optuna.create_study(direction='maximize')
        # study.optimize(lambda trial: objective_lgb(trial, x_train, target), n_trials=100)
        study.optimize(lambda trial: objective_lgb(trial, x_train, target), timeout=100) # degubのため時間制限
        bestparams = study.best_trial.params

    # ベストパラメータを出力
    print("params:", study.best_params)
    print("best value: ", study.best_value)
    
    return bestparams


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


def ht_xgb(train):
    features_col = ["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]
    x_train = train[features_col].values
    target = train["Survived"].values
    
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
    clf = GridSearchCV(xgb.XGBClassifier(), ht_param, cv=5, scoring="accuracy")
    clf.fit(x_train, target)
    print("最良パラメータ: {}".format(clf.best_params_))
    print("最良交差検証スコア: {:.2f}".format(clf.best_score_))
    
    sys.exit()
    
    return 


def xgb_model(train, test, best_params):
    features_col = ["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]
    x_train = train[features_col].values
    target = train["Survived"].values
    x_test = test[features_col].values

    # train
    xgb_tree = xgb.XGBClassifier(**best_params)
    xgb_tree.fit(x_train, target)

    train_prediction_prob = xgb_tree.predict(x_train)
    test_prediction_prob = xgb_tree.predict(x_test)
    
    alfa = find_best_alfa(target, train_prediction_prob)
    
    train_prediction = np.where(train_prediction_prob > alfa, 1, 0)
    test_prediction = np.where(test_prediction_prob > alfa, 1, 0)

    return train_prediction, test_prediction


def lgb_model(train, test, best_params):
    features_col = ["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]
    x_train = train[features_col].values
    target = train["Survived"].values
    x_test = test[features_col].values

    lgb_train = lgb.Dataset(x_train, label=target)

    # train
    model = lgb.train(train_set=lgb_train, **best_params)

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
    
    with redirect_stderr(open(os.devnull, 'w')):
        train, test = data_preprocessing(train, test)
    
    with redirect_stdout(open(os.devnull, 'w')):
        best_params = use_optuna_for_xgb(train)
    train_prediction, test_prediction = xgb_model(train, test, best_params)
    
    with redirect_stdout(open(os.devnull, 'w')):
        best_params = use_optuna_for_lgb(train)
    print(best_params)
    train_prediction, test_prediction = lgb_model(train, test, best_params)
    
    output_result(train, test, train_prediction, test_prediction)
    
    return

if __name__ == "__main__":
    main()